工程地址：https://github.com/huggingface/transformers

文档地址：https://huggingface.co/transformers/



[TOC]

## 安装

```shell
pip install transformers==3.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install pip install spacy-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```





## 预训练模型下载



```shell
mkdir /sdb/tmp/pretrained_model/transformers
cd /sdb/tmp/pretrained_model/transformers

## bert-base-uncased
mkdir bert-base-uncased
cd bert-base-uncased
wget -c -t 0 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin -O pytorch_model.bin

wget -c -t 0 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json -O config.json

wget -c https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt -O vocab.txt


## bart-base
mkdir bart-base
cd bart-base
wget -c -t 0 https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-base/config.json -O config.json

wget -c -t 0 https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-base/pytorch_model.bin -O pytorch_model.bin

## bart-large-cnn cdn.huggingface.co速度更快
mkdir bart-large-cnn
cd bart-large-cnn
wget -c -t 0 https://cdn.huggingface.co/facebook/bart-large-cnn/pytorch_model.bin
wget -c -t 0 https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-cnn/config.json
wget -c -t 0 https://cdn.huggingface.co/facebook/bart-large-cnn/vocab.json
wget -c -t 0 https://cdn.huggingface.co/facebook/bart-large-cnn/merges.txt
```







```python
BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    'bert-base-german-dbmdz-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    'bert-base-german-dbmdz-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
}
```



```python
BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-config.json",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-config.json",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-config.json",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-config.json",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-config.json",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-config.json",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-config.json",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-config.json",
    'bert-base-german-dbmdz-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-config.json",
    'bert-base-german-dbmdz-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-config.json",
}
```





## 测试

### umask

```python
from transformers import BertModel,BertTokenizer,BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained("/sdb/tmp/pretrained_model/transformers/bert-base-uncased")
model = BertForMaskedLM.from_pretrained("/sdb/tmp/pretrained_model/transformers/bert-base-uncased")
unmasker = pipeline('fill-mask', model=model,tokenizer=tokenizer)
unmasker("Hello I'm a [MASK] model.")
```



### summary

```python
from transformers import BartModel,BartTokenizer,BartForConditionalGeneration
tokenizer = BartTokenizer.from_pretrained("/sdb/tmp/pretrained_model/transformers/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("/sdb/tmp/pretrained_model/transformers/bart-large-cnn/")
summarizer = pipeline('summarization', model=model,tokenizer=tokenizer)

ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""

summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False)
```





### GPT2 分类

```shell
## bert-base-uncased
mkdir gpt2-medium
cd gpt2-medium
wget -c -t 0 https://cdn-lfs.huggingface.co/gpt2/7c5d3f4b8b76583b422fcb9189ad6c89d5d97a094541ce8932dce3ecabde1421 -O pytorch_model.bin
wget -c -t 0 https://huggingface.co/gpt2/resolve/main/config.json -O config.json

wget -c https://huggingface.co/gpt2/resolve/main/vocab.json -O vocab.txt
```



