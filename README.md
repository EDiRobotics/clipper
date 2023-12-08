## Bind SSH-FS
Mount the remote dataset folder to local
```bash
bash bind_fs.sh
```


## Train
Run the training script
```bash
python train_clip.py --save-frequency 1 --zeroshot-frequency 1 --dataset-type "synthetic" --train-num-samples 16 --warmup 1 --batch-size 4 --lr 1e-3 --wd 0.1 --epochs 1 --workers 2 --model ViT-B-32
```
