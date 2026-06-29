# Run Commands for All Datasets


---

## 1. Animal 

```bash
python main.py --i_d 12 --missrate 0.5 --seed 42 \
    --pretrain_epochs 200 --FineTuning_epochs 100 \
    --lr_pre 0.0005 --lr_finetuning 0.00001 \
    --para_loss 1.6e-05 5 5 10 0.3 0.7
```

---

## 2. ALOI_100 
```bash
python main.py --i_d 3 --missrate 0.3 --seed 12 \
    --pretrain_epochs 200 --FineTuning_epochs 100 \
    --lr_pre 0.0005 --lr_finetuning 0.00001 \
    --para_loss 1.6e-06 1 1 10 0.1 0.9
    --tau_w 0.07
```

---

## 3. Digit4k Dataset

```bash
python main.py --i_d 20 --missrate 0.5 --seed 42 \
    --pretrain_epochs 200 --FineTuning_epochs 100 \
    --lr_pre 0.0005 --lr_finetuning 0.00001 \
    --para_loss 1.6e-05 5 10 50 0.2 0.8
```

---

## 4. 100Leaves Dataset

```bash
python main.py --i_d 8 --missrate 0.5 --seed 42 \
    --pretrain_epochs 200 --FineTuning_epochs 100 \
    --lr_pre 0.0005 --lr_finetuning 0.00001 \
    --para_loss 1.6e-05 5 5 200 0.3 0.7
```

---

## 5. Reuters_21578 Dataset

```bash
python main.py --i_d 34 --missrate 0.5 --seed 42 \
    --pretrain_epochs 200 --FineTuning_epochs 100 \
    --lr_pre 0.0005 --lr_finetuning 0.00001 \
    --para_loss 1.6e-05 5 5 10 0.3 0.7
```

---

## 6. VGGFace2-50 Dataset

```bash
python main.py --i_d 39 --missrate 0.5 --seed 42 \
    --pretrain_epochs 200 --FineTuning_epochs 100 \
    --lr_pre 0.0005 --lr_finetuning 0.00001 \
    --para_loss 1.6e-05 5 5 10 0.3 0.7
```

---

