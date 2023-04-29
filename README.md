# Team Weekend Warriors: CS753 Project

## Members:
### Aditya Jain
### Govind Saju
### Sankalp Parashar

###  Diff of the code that generated our result (from colab.py) vs their code is in diff.txt. Reorganized the structure of repo for convenience. The logs are now in Logfiles folder, Plots are in Plots folder and Results are in Results_txt folder.

## Instructions to run:
* Install the required modules: pip install -r requirements.txt
* To run the python file from our notebook code: python colab.py (no arguments here)
* Run the bash script with the mask fraction parameter: script original .sh <mask_fraction> 

The bash sript to run the code is 'script original.sh'. It requires the mask fraction as a parameter.

* We tried running their code first. There were issues with loss not converging after many epochs.
* Following this, we followed online resources and tutorials (links of which we have added below) to rewrite our own version, which worked well right away. 
* The plots generated for comparison between WCTC and normal CTC are in the png files with self explanatory names. 

Our work: 
* Setting up the Wav2vec2 pipeline, with pretrained feature extractor and tokenizer
* Finetuning the model with our WCTC loss as well as with normal CTC loss, and comparing the results
* Obtaining stable training of our implementation, and marked difference between the performances of WCTC loss and normal CTC loss on this task. 

File descriptions:
* wctc.py: Contains the code for the wctc loss
* model.py: Contains the Wav2vec2 model, with the forward function redefined to include WCTC
* classes.py: Contains classes to preprocess data/setup the pipeline
* main.py The main file to run the code, evaluate and save the model

The results plots are present in the repository. Logfiles showing predicted sentences have also been uploaded.

Structure of modified DP:
![image](https://user-images.githubusercontent.com/81635126/234790105-25574c8a-d6f5-4c03-a5f3-0c88643ce477.png)


Reference tutorials and source codes (for normal implementation without WCTC): https://huggingface.co/blog/fine-tune-wav2vec2-english

https://huggingface.co/docs/transformers/model_doc/wav2vec2

https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1616

WCTC Paper link: https://openreview.net/pdf?id=0RqDp8FCW5Z


