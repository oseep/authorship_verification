This directory contains the experiments for generating fanfictions using GPT2 and applying the authorship verification model. 

Data Needed:
Pan Dataset: Download from here or available in the PSAL computer at `/media/disk1/social/authoorship_verification/data/pan/`
Preprocessed PAN dataset: Use the script in `pan/preprocess_hpc.py` and run it on NYU green or copy from the PSAL computer at `/media/disk1/social/authoorship_verification/temp_data/pan/`



File Overviews
===
`generate_ai_text_hpc.py`: 
  This file is meant to by run as an NYU Greene batch job. It loads already preprocessed fanfictions, and writes a series of `human_ai_preprocessed<id>.jsonl` files.
  The `generate_ai_and_human_text_pair` function takes in a preprocessed document, chunks it, and uses every other chunk to use it as a prompt to generate text using GPT2.
  It concatenates these chunks to create an "AI-generated" version of the fanfiction. The output file contains json objects in which is line if of the format:
  ```
  {
      'id': Original dataset record id,
      'fandoms': Original fandoms,
      'pair': [ // Document pair
          {'human': The human version, 'ai': The AI-generated version},
          {'human': The human version, 'ai': The AI-generated version},
      ]
  }
  ```   
  `generate_ai_text_hpc_2.py` is similar, it just appends to the `human_ai_preprocessed<id>.jsonl`. This can be used if your initial run crashes halfway to save time.
  This same script can be used to generate text using other models by loading the fine-tuned model and tokenizer in lines 72 & 73.
  
  
`AI_vs_Human_AV.ipynb`: This is the notebook that runs the AV model on the generated texts. It was used to generate the stats and plots in my thesis. 
Download the trained AV model for fanfictions from https://github.com/janithnw/pan2021_authorship_verification/tree/main/temp_data/large_model_training_data or any similar model

`explore.ipynb`: This notebook is my initial attemp at trying to finetune the GPT2 model. Later I moved the code to `fine_tune_fanfics.py`. The first couple of blocks tokenize the data 
using the GPT2 tokenizer and genarates `input_ids.npy`, `attention_mask.npy`, and `metadata.p`. This is needed by `fine_tune_fanfics.py` script as well. The next couple of blocks does the actual fine tuning.
However, I realized doing that part in a script was more robust. I used https://wandb.ai/ do monitor the training progress and monitor loss curves. If you decide to use it, create an account and log in.
You should get a prompt to login in `fine_tune_fanfics.py`, Line 43. 

The rest of the blocks in `explore.ipynb` can be used to load a fine-tuned model and generate some sample texts. This is where I stopped. The idea would be to fine tune the model 
completely and perform an analysis simlar to that of `AI_vs_Human_AV.ipynb` using the fine tuned model.

