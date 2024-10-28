import torch
from huggingsound import SpeechRecognitionModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ASR:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #audio_recognition_model_path = "E:/shanghainese_model/spycsh/shanghainese-wav2vec-3800"
        # audio_recognition_model_path = "E:/shanghainese_model/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
        audio_recognition_model_path = "E:/shanghainese_model/TingChen-ppmc/whisper-small-Shanghai"
        self.model = SpeechRecognitionModel(audio_recognition_model_path, device=device)
        self.tokenizer = self.model.processor.tokenizer

        mt_model_path = 'E:/shanghainese_model/shanghainese-opus-sh-zh-3500'
        self.mt_tokenizer = AutoTokenizer.from_pretrained(mt_model_path)
        self.mt_model = AutoModelForSeq2SeqLM.from_pretrained(mt_model_path, local_files_only=True).to(device)
        
    def transcribe(self, wav):
        audio_path = [wav]
        res = self.model.transcribe(audio_path, batch_size=1)[0]
        transcription = res['transcription']
        probabilities = res['probabilities']
        return {"transcription": transcription, "transcription_score": probabilities}

    def generate_translation(self, model, tokenizer, example):
        """print out the source, target and predicted raw text."""
        input_ids = example['input_ids']
        input_ids = torch.LongTensor(input_ids).view(1, -1).to(model.device)
        # print('input_ids: ', input_ids)
        generated_ids = model.generate(input_ids, max_new_tokens=64)
        # print('generated_ids: ', generated_ids)
        prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return prediction

    def translation(self, text: str):
        with self.mt_tokenizer.as_target_tokenizer():
            model_inputs = self.mt_tokenizer(text, max_length=64, truncation=True)
            example = {}
            example['sh'] = text
            example['input_ids'] = model_inputs['input_ids']
            translation = self.generate_translation(model=self.mt_model, tokenizer=self.mt_tokenizer, example=example)
        return translation