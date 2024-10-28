
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#mt_model_path = 'E:/shanghainese_model/opus-mt-en-zh'
mt_model_path = 'E:/shanghainese_model/shanghainese-opus-sh-zh-3500'
tokenizer = AutoTokenizer.from_pretrained(mt_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(mt_model_path)

text = "侬讲呃今朝要来呃！"
batch = tokenizer.prepare_seq2seq_batch(src_texts=[text])

# Make sure that the tokenized text does not exceed the maximum
# allowed size of 512
batch["input_ids"] = batch["input_ids"][:, :512]
batch["attention_mask"] = batch["attention_mask"][:, :512]

# Perform the translation and decode the output
translation = model.generate(**batch)
result = tokenizer.batch_decode(translation, skip_special_tokens=True)
print(result)