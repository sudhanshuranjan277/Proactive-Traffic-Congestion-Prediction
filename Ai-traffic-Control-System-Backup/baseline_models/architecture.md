             


                                        ┌─────────────────────────┐
                           │     Research Paper      │
                           │  + Appendix + Equations │
                           └────────────┬────────────┘
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │ Implementation Blueprint│
                           │ (Modules & Data Flow)   │
                           └────────────┬────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────┐
                    │ Configuration (No Hardcoding)      │
                    └────────────┬───────────────────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ Dataset Generation │
                       └─────────┬──────────┘
                                 │
                                 ▼
                ┌────────────────────────────────┐
                │ Raw Dataset (METR-LA/PEMS-BAY) │
                └──────────────┬─────────────────┘
                               │
                               ▼
                       Data Validation
                               │
                               ▼
                       Data Preprocessing
                               │
                               ▼
                       Window Generation
                               │
                               ▼
                       Graph Generation
                               │
                               ▼
                      Processed Dataset
                               │
                               ▼
                         Model Pipeline
                               │
                               ▼
        ┌─────────────────────────────────────────────┐
        │  
               TSFormer                                    │
        │    KnowledgeDistillation     
                     │
        
        │      Structure Generato  
                    │
                   (GTS)                  
                    │
                    │ 
               GraphWaveNet                                │
        └─────────────────────────────────────────────┘
                               │
                               ▼
                           Training
                               │
                               ▼
                          Evaluation
                               │
                               ▼
                      Reproduced Results