from gtts import gTTS
import pygame
import io
import threading
import time
import os

class SpeechEngine:
    def __init__(self):
        pygame.mixer.init()
        self.is_speaking = False
        self.last_speech_time = 0
        self.speech_cooldown = 10 
        
    def text_to_speech(self, text, lang='en'):
        """Convert text to speech and play it"""
        if self.is_speaking or (time.time() - self.last_speech_time) < self.speech_cooldown:
            return
            
        self.is_speaking = True
        
        def speak():
            try:
           
                tts = gTTS(text=text, lang=lang, slow=False)
                
                
                mp3_fp = io.BytesIO()
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                
               
                pygame.mixer.music.load(mp3_fp)
                pygame.mixer.music.play()
                
               
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                    
            except Exception as e:
                print(f"Speech error: {e}")
            finally:
                self.is_speaking = False
                self.last_speech_time = time.time()
        
        
        thread = threading.Thread(target=speak)
        thread.daemon = True
        thread.start()
    
    def stop_speech(self):
        """Stop current speech"""
        pygame.mixer.music.stop()
        self.is_speaking = False