from typing import Any, Dict, Optional
from pythonosc import udp_client
from .common import GameResult

class UnrealClient:
    def __init__(self, ip: str = "127.0.0.1", port: int = 7000):
        self.client = udp_client.SimpleUDPClient(ip, port)

    def send_game_result(self, result: GameResult) -> None:
        """Sends the game result data to Unreal Engine via OSC."""
        
        # Send Chord Data
        if result.chord:
            self.client.send_message("/music/chord/label", result.chord.label)
            # You might want to send individual notes if available in the future
        
        # Send Emotion Data
        if result.emotion:
            self.client.send_message("/music/emotion/label", result.emotion.label)
            self.client.send_message("/music/emotion/confidence", float(result.emotion.confidence))
            
            # Send probabilities as a list or individual messages
            for emotion, prob in result.emotion.probabilities.items():
                self.client.send_message(f"/music/emotion/probability/{emotion}", float(prob))

        # Send Dialogue/LLM Result
        if result.dialogue:
            self.client.send_message("/music/llm/content", result.dialogue.content)
            self.client.send_message("/music/llm/role", result.dialogue.role)

        # Send Descriptors (Audio Features)
        for key, value in result.descriptors.items():
            # OSC supports float, int, string, blob. Ensure value is compatible.
            if isinstance(value, (float, int, str)):
                self.client.send_message(f"/music/descriptor/{key}", value)
            
        print(f"Sent GameResult to Unreal Engine at {self.client._address}:{self.client._port}")
