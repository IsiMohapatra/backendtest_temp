"""
Fixed automotive agent with proper async lifecycle management.
Prevents "Task was destroyed but it is pending!" errors.
"""
 
import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, RoomInputOptions, Agent
from tools.vision_capabilities import VisionCapabilities
# Lazy import to avoid import errors when virtual environment is not activated
# from workflows.diagnostic_workflow import AsyncDiagnosticAgent
 
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
 
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")
 
 
def load_prompts(agent: str) -> dict:
    """Load prompts for a given agent from the registry."""
    prompts_file = BASE_DIR / "prompts" / "prompts.json"
    with open(prompts_file, "r", encoding="utf-8") as f:
        registry = json.load(f)
    cfg = registry[agent]
    result = {}
    if "instructions_file" in cfg:
        instructions_path = BASE_DIR / cfg["instructions_file"]
        with open(instructions_path, "r", encoding="utf-8") as f:
            result["instructions"] = f.read()
    if "greetings_file" in cfg:
        greetings_path = BASE_DIR / cfg["greetings_file"]
        with open(greetings_path, "r", encoding="utf-8") as f:
            result["greetings"] = json.load(f)
    return result
 
 
class Assistant(VisionCapabilities, Agent):
    """Automotive assistant with improved async handling."""

    def __init__(self, vehicle: str = None, model: str = None) -> None:
        prompts = load_prompts("automotive")
        Agent.__init__(self, instructions=prompts["instructions"])
        VisionCapabilities.__init__(self)
        self.greetings = prompts.get("greetings", {})
        self.diagnostic_agent = None
        self.vehicle = vehicle
        self.model = model
    async def initialize_diagnostic_agent(self):
        """Initialize the diagnostic agent asynchronously."""
        if self.diagnostic_agent is None:
            # Lazy import to avoid import errors when virtual environment is not activated
            try:
                from workflows.diagnostic_workflow import AsyncDiagnosticAgent
                self.diagnostic_agent = AsyncDiagnosticAgent(
                    vehicle=self.vehicle, 
                    model=self.model
                )
                await self.diagnostic_agent.initialize()
            except ImportError as e:
                logger.error(f"Failed to import AsyncDiagnosticAgent: {e}")
                logger.error("Make sure the virtual environment is activated with: .\\allion\\Scripts\\activate")
                raise
    async def cleanup_diagnostic_agent(self):
        """Clean up diagnostic agent resources."""
        if self.diagnostic_agent:
            await self.diagnostic_agent.cleanup()
            self.diagnostic_agent = None
    def get_greeting(self, lang: str = "en") -> str:
        """Return a greeting string for the given language."""
        return self.greetings.get(lang, self.greetings.get("en", ""))
    async def handle_user_message(self, message: str) -> str:
        """Process user message through diagnostic agent."""
        await self.initialize_diagnostic_agent()
        return await self.diagnostic_agent.chat(message)
 
 
def get_simplified_config(lang_code: str, voice_base: str, vehicle: str = None, model: str = None):
    """Simplified config selection with vehicle information."""
    from configs.language_config import get_config_for_language
    return get_config_for_language(lang_code, voice_base, vehicle=vehicle, model=model)
 
 
async def entrypoint(ctx: agents.JobContext) -> None:
    """
    Improved entrypoint with proper async resource management.
    """
    await ctx.connect()
    # Configuration logic (simplified)
    cfg_box = {
        "language": os.getenv("AGENT_LANG", "en").lower(),
        "voiceBase": os.getenv("AGENT_VOICEBASE", "Voice Assistant"),
        "vehicle": None,
        "model": None,
    }
    got_config = asyncio.Event()
 
    def _try_set_from_metadata(md: Optional[str]):
        """Parse metadata and update config."""
        try:
            if not md or not isinstance(md, (str, bytes, bytearray)):
                return
            payload = json.loads(md)
            if "language" in payload and payload["language"] in ("en", "hi", "kn"):
                cfg_box["language"] = payload["language"]
                got_config.set()
            if "voiceBase" in payload and payload["voiceBase"] in ("Voice Assistant", "Live Assistant"):
                cfg_box["voiceBase"] = payload["voiceBase"]
                got_config.set()
            if "vehicle" in payload and payload["vehicle"]:
                cfg_box["vehicle"] = payload["vehicle"]
                logger.info(f"Vehicle information received: {payload['vehicle']}")
                got_config.set()
            if "model" in payload and payload["model"]:
                cfg_box["model"] = payload["model"]
                logger.info(f"Model information received: {payload['model']}")
                got_config.set()
        except Exception as e:
            logger.warning(f"Metadata parse error: {e}")
 
    # Metadata handling (simplified)
    if ctx.room.metadata:
        _try_set_from_metadata(ctx.room.metadata)
 
    for p in ctx.room.remote_participants.values():
        _try_set_from_metadata(getattr(p, "metadata", None))
        if hasattr(p, "on"):
            @p.on("metadata_changed")
            def _on_meta_changed(p_local=p):
                _try_set_from_metadata(p_local.metadata)
 
    @ctx.room.on("participant_connected")
    def _on_participant_connected(p):
        _try_set_from_metadata(getattr(p, "metadata", None))
        if hasattr(p, "on"):
            @p.on("metadata_changed")
            def _on_meta_changed(p_local=p):
                _try_set_from_metadata(p_local.metadata)
 
    # Wait for config or timeout
    try:
        await asyncio.wait_for(got_config.wait(), timeout=8.0)
    except asyncio.TimeoutError:
        logger.info("Using default configuration")
 
    # Get configuration
    lang = cfg_box["language"]
    voice_base = cfg_box["voiceBase"]
    vehicle = cfg_box["vehicle"]
    model = cfg_box["model"]
    logger.info(f"Using config: language={lang}, voiceBase={voice_base}, vehicle={vehicle}, model={model}")
    config_dict = get_simplified_config(lang, voice_base, vehicle=vehicle, model=model)
    
    # Create session and assistant
    session = AgentSession(**config_dict)
    
    # ✅ NEW: Set room reference on DiagnosticLLMAdapter if it exists
    if "llm" in config_dict and hasattr(config_dict["llm"], "room"):
        config_dict["llm"].room = ctx.room
        logger.info("✅ Set room reference on DiagnosticLLMAdapter for data channel publishing")
    
    assistant = Assistant(vehicle=vehicle, model=model)
    try:
        # Start session
        await session.start(
            room=ctx.room,
            agent=assistant,
            room_input_options=RoomInputOptions(video_enabled=True),
        )
        # Send initial greeting
        greeting = assistant.get_greeting(lang)
        logger.info(f"Sending greeting via session.generate_reply: {greeting}")
        await session.generate_reply(instructions=greeting)
        logger.info("✅ Greeting sent via session")
        
        # Keep the session running until the room is disconnected
        # Note: session.aclose() will be called automatically when the context exits
        logger.info("Session started, waiting for room events...")
        
        # Wait for the room to be disconnected or until we receive a shutdown signal
        disconnected = asyncio.Event()
        
        @ctx.room.on("disconnected")
        def on_room_disconnected():
            logger.info("Room disconnected")
            disconnected.set()
        
        # Wait until the room is disconnected
        await disconnected.wait()
    finally:
        # CRITICAL: Clean up diagnostic agent resources
        await assistant.cleanup_diagnostic_agent()
        logger.info("Agent cleanup completed")
 
 
def run_agent():
    """Run the agent with proper error handling."""
    try:
        agents.cli.run_app(
            agents.WorkerOptions(
                entrypoint_fnc=entrypoint,
                initialize_process_timeout=60.0,
                shutdown_process_timeout=60.0,
                job_memory_warn_mb=15000,
                use_separate_process=False,
            )
        )
    except TypeError:
        # Fallback for different library versions
        os.environ["LIVEKIT_AGENTS_DISABLE_SEPARATE_PROCESS"] = "1"
        agents.cli.run_app(
            agents.WorkerOptions(
                entrypoint_fnc=entrypoint,
                initialize_process_timeout=60.0,
                shutdown_process_timeout=60.0,
                job_memory_warn_mb=15000,
            )
        )