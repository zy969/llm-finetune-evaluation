# Fine-tuning LLMs: Base vs. Full vs. LoRA

Task-oriented evaluation of LLaMA-2-7B under three adaptation strategies:
- **Base**: original pretrained model  
- **Full fine-tuned**: instruction-tuned, all parameters updated  
- **LoRA fine-tuned**: parameter-efficient adaptation  

### Tasks
- Extractive QA (SQuAD, TriviaQA)  
- Abstractive Summarization (CNN/DailyMail, Gigaword)  
- Instruction Following (AlpacaEval, GPTeacher)  

### Methods
- Metrics: EM/F1 (QA), ROUGE (summarization), GPT-4 preference (instruction following)  
- Statistical analysis: ANOVA, Wilcoxon tests, effect sizes  

### Focus
Accuracy gains, behavioral alignment, and efficiency trade-offs across fine-tuning strategies.
