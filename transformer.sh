uv run models/transformer/train.py
if [ $? -ne 0 ]; then
    echo "Error in train.py. Exiting..."
    exit 1
fi

uv run models/transformer/train_cosine.py
if [ $? -ne 0 ]; then
    echo "Error in train_cosine.py. Exiting..."
    exit 1
fi

uv run models/transformer/test.py
if [ $? -ne 0 ]; then
    echo "Error in test.py. Exiting..."
    exit 1
fi

echo "All scripts ran successfully!"
