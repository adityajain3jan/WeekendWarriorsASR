# Team Weekend Warriors: CS753 Project

## Members:
### Aditya Jain
### Govind Saju
### Sankalp Parashar

![image](https://user-images.githubusercontent.com/81635126/234790105-25574c8a-d6f5-4c03-a5f3-0c88643ce477.png)


The bash sript to run the code is 'script original.sh'. It requires the mask fraction as a parameter.

Our work includes:
* Preprocessing the TIMIT dataset (including mask application)
* Setting up the entire Wav2vec2 pipeline, with pretrained feature extractor and tokenizer
* Finetuning the model with our WCTC loss as well as with normal CTC loss, and comparing the results
* Obtaining stable training of our implementation, and marked difference between the performances of WCTC loss and normal CTC loss on this task. 

File descriptions:
* wctc.py: Contains the code for the wctc loss
* model.py: Contains the Wav2vec2 model, with the forward function redefined to include WCTC
* classes.py: Contains classes to preprocess data/setup the pipeline
* main.py The main file to run the code, evaluate and save the model

The results plots are present in the repository. Logfiles showing predicted sentences have also been uploaded.

## Instructions to run:
* Install the required modules: pip install -r requirements.txt
* Run the bash script with the mask fraction parameter: script original .sh <mask_fraction> 

Reference tutorial (for normal implementation without WCTC): https://huggingface.co/blog/fine-tune-wav2vec2-english

WCTC Paper link: https://openreview.net/pdf?id=0RqDp8FCW5Z
