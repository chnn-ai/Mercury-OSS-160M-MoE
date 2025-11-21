<img width="1024" height="439" alt="Mercury logo" src="https://github.com/user-attachments/assets/4acb7516-086c-4684-9578-76cf70d9f835" />


This project is about creating the most efficient Small Language Model (SLM) with reasoning capabilies. I uploaded the training and model code for the first version of Mercury.
This model is a 160M parameters, Group Query Attention + Mixture of Experts. The data used is https://huggingface.co/datasets/thesouth/my-fineweb-tokenized which is a tokenized version 
of https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle, I only took part of that dataset (350k docs) to run small ablations before training the model on the whole dataset.
Unfortunately, I was not able to run those ablations because I have no compute budget. The code uploaded is only about pretraining, I plan to add Muon Optimizer, decrease the memory footprint
during training which is (unsurprisingly) enormous due to the MoE implementation and then dive into the RL part.
