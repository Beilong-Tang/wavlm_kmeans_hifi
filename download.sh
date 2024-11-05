mkdir -p ckpt

echo "Downloading hifi-ckpt in ckpt folder:"
wget -P ckpt https://github.com/bshall/knn-vc/releases/download/v0.1/g_02500000.pt
# echo "Downloading detokenizer conformer in ckpt folder:"
# wget -P ckpt https://github.com/bshall/knn-vc/releases/download/v0.1/g_02500000.pt


