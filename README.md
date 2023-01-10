# project-OBA

Mo Shams <MShamsCBR.gmail.com>  
Jan 10, 2023

To investigate the spatial profile of the object-based attention.

---

### Overview
Q1: Does attention to an instance of an object trasnfer to the same 
instance of the object across space?

Q2: Does attention to an instance of an object transfer to other instances 
of the object across space?

Q3: Does attention to an instance of an object transfer to other images 
typically (or artificially by training) associated to that object across space?

---

### Required Packages
For stimulus generation:
- Python 3.8.13
- Psychopy 2022.1.4

For data analysis:
- MNE 1.3.0
- Python 3.9.15

---

### Directory Organization
```
OBA
|   .gitignore
|   figures.ai
|   figXX.eps
|   figXX.[script]
|   README.md
|__ analysis
|       aXXX_[name].[script]
|__ data
|   |   aXXX_[name].[data]
|   \__ raw
|           *.[data]
|           recording_notes.txt   
|__ docs
|       *.[text]
|__ lib
|       *.[script]
|__ results
|       *.pdf
|       *.key
\__ stimulus
    |   expXX_[name].[script]
    |   test_[name].[script]
    \__ image
        \__ source
```
---

### Pipeline
            