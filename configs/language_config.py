"""
Simplified language configurations that use the improved async diagnostic agent
with enforced target-language and proper TTS in Live Assistant mode.
"""

import os
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import silero, cartesia, deepgram, google, sarvam, openai
from workflows.diagnostic_workflow import DiagnosticLLMAdapter

# Shared components
COMMON_CONFIG = {
    "vad": silero.VAD.load(),
    "turn_detection": MultilingualModel(),
    "use_tts_aligned_transcript": True,
}

class LanguageConfig:
    """Base configuration class for all languages."""
    @staticmethod
    def get_diagnostic_llm(language: str = "en", room=None, vehicle: str = None, model: str = None):
        """Get the diagnostic agent wrapped as LLM adapter, with language enforced."""
        return DiagnosticLLMAdapter(target_language=language, room=room, vehicle=vehicle, model=model)

    @classmethod
    def get_base_config(cls):
        """Get base configuration common to all languages."""
        return COMMON_CONFIG.copy()

class EnglishConfig(LanguageConfig):
    """English voice assistant configuration."""
    @classmethod
    def get_config(cls, voice_base: str = "Voice Assistant", vehicle: str = None, model: str = None):
        config = cls.get_base_config()
        if voice_base == "Live Assistant":
            config.update({
                "llm": google.beta.realtime.RealtimeModel(
                    model="gemini-2.0-flash-exp",
                    voice="Puck",
                    temperature=0.5,
                ),
                "stt": deepgram.STT(model="nova-3", language="multi"),
                "tts": cartesia.TTS(
                    model="sonic-2",
                    voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
                ),
            })
        else:
            # Voice Assistant mode: Use LangGraph workflow with RAG capabilities
            config.update({
                "llm": cls.get_diagnostic_llm(language="en", vehicle=vehicle, model=model),
                "stt": deepgram.STT(model="nova-3", language="multi"),
                "tts": cartesia.TTS(
                    model="sonic-2",
                    voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
                ),
            })
        return config

class HindiConfig(LanguageConfig):
    """Hindi voice assistant configuration."""
    @classmethod
    def get_config(cls, voice_base: str = "Voice Assistant", vehicle: str = None, model: str = None):
        config = cls.get_base_config()
        if voice_base == "Live Assistant":
            config.update({
                "llm": google.beta.realtime.RealtimeModel(
                    model="gemini-2.0-flash-exp",
                    voice="Puck",
                    temperature=0.5,
                    language="hi-IN",
                ),
                # Ensure TTS/STT are present in Live Assistant mode too
                "stt": sarvam.STT(model="saarika:v2.5", language="hi-IN"),
                "tts": sarvam.TTS(
                    target_language_code="hi-IN",
                    speaker="abhilash",
                    api_key=os.getenv("SARVAM_API_KEY"),
                ),
            })
        else:
            config.update({
                "llm": cls.get_diagnostic_llm(language="hi", vehicle=vehicle, model=model),
                "stt": sarvam.STT(model="saarika:v2.5", language="hi-IN"),
                "tts": sarvam.TTS(
                    target_language_code="hi-IN",
                    speaker="abhilash",
                    api_key=os.getenv("SARVAM_API_KEY"),
                ),
            })
        return config

class KannadaConfig(LanguageConfig):
    """Kannada voice assistant configuration."""
    @classmethod  
    def get_config(cls, voice_base: str = "Voice Assistant", vehicle: str = None, model: str = None):
        config = cls.get_base_config()
        if voice_base == "Live Assistant":
            config.update({
                "llm": google.beta.realtime.RealtimeModel(
                    model="gemini-2.0-flash-exp",
                    voice="Puck",
                    temperature=0.5,
                    language="kn-IN",
                ),
                # Ensure TTS/STT are present in Live Assistant mode too
                "stt": sarvam.STT(model="saarika:v2.5", language="kn-IN"),
                "tts": sarvam.TTS(
                    target_language_code="kn-IN",
                    speaker="abhilash",
                    api_key=os.getenv("SARVAM_API_KEY"),
                ),
            })
        else:
            config.update({
                "llm": cls.get_diagnostic_llm(language="kn", vehicle=vehicle, model=model),
                "stt": sarvam.STT(model="saarika:v2.5", language="kn-IN"),
                "tts": sarvam.TTS(
                    target_language_code="kn-IN",
                    speaker="abhilash",
                    api_key=os.getenv("SARVAM_API_KEY"),
                ),
            })
        return config

# Factory function for backwards compatibility
def get_config_for_language(language_code: str, voice_base: str = "Voice Assistant", vehicle: str = None, model: str = None):
    """Factory function to get configuration for a specific language and vehicle info."""
    config_map = {
        "en": EnglishConfig,
        "hi": HindiConfig, 
        "kn": KannadaConfig,
    }
    config_class = config_map.get(language_code, EnglishConfig)
    return config_class.get_config(voice_base, vehicle=vehicle, model=model)
