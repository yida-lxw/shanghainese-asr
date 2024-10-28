from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mt_model_path = 'E:/shanghainese_model/shanghainese-opus-sh-zh-3500'
tokenizer = AutoTokenizer.from_pretrained(mt_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(mt_model_path).to(device)

def generate_translation(model, tokenizer, example, max_new_tokens:int=64):
    input_ids = example['input_ids']
    input_ids = torch.LongTensor(input_ids).view(1, -1).to(model.device)
    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return prediction

def translate2Shanghainese(text:str, chunk_size:int=128):
    with tokenizer.as_target_tokenizer():
        model_inputs = tokenizer(text, max_length=chunk_size, truncation=True)
        example = {}
        example['input_ids'] = model_inputs['input_ids']
        return generate_translation(model, tokenizer, example, max_new_tokens=chunk_size)


def translate_shanghainese_to_mandarin(shanghainese_text:str, chunk_size:int=64):
    try:
        if len(shanghainese_text) <= chunk_size:
            prediction_result = translate2Shanghainese(shanghainese_text, chunk_size)
        else:
            chunks = [shanghainese_text[i:i + chunk_size] for i in range(0, len(shanghainese_text), chunk_size)]
            prediction_array = []
            for per_text in chunks:
                prediction = translate2Shanghainese(per_text, chunk_size)
                prediction_array.append(prediction)
            prediction_result = "\n".join(prediction_array)
        return {"code": 200, "msg": "success", "translation": prediction_result}
    except:
        return {"code": 500, "msg": "failed", "translation": ""}
