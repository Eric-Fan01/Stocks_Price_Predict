import argparse
from CBAmodel import CNNBiLSTMAMModel
from utils import save_model, load_model, get_data_preprocessor
import pickle
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model_path', default='./checkpoints/autosave') # will judge the format by file extension
    parser.add_argument('--time_steps', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--save_format', choices=['h5', 'pb', 'weights'], default='pb', help='Format to save model (h5(weight only) or pb).')
    return parser.parse_args()

def main(args):
    if args.mode == 'train':
        print("🔁  Preprocessing...")
        # load data preprocessor based on file extension
        preprocessor = get_data_preprocessor(args.data, time_steps=args.time_steps)
        X, y = preprocessor.preprocess()

        print("🚀  Building model...")
        model = CNNBiLSTMAMModel(time_steps=args.time_steps, num_features=X.shape[2])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss='mae', metrics=['mse'])

        print("🏋️  Training...")
        model.fit(X, y, batch_size=args.batch_size, epochs=args.epochs)

        save_model(model, args.model_path, format=args.save_format)

    elif args.mode == 'test':
        model = load_model(args.model_path)
        preprocessor = get_data_preprocessor(args.data, time_steps=args.time_steps)
        X, y = preprocessor.preprocess()

        print("🧪  Evaluating...")
        loss, mse = model.evaluate(X, y)
        print(f"📊  MAE: {loss:.4f} | MSE: {mse:.4f}")

        y_pred = model.predict(X)



        with open("data/aapl_norm.pkl", "rb") as f:
            norm = pickle.load(f)
        mean = norm['Closing price']['mean']
        std = norm['Closing price']['std']
        # 还原 y_pred 和 y
        y_pred_real = y_pred.flatten() * std + mean
        y_real = y * std + mean

        for i in range(10):
            print(f"[{i}]  True: {y_real[i]:.2f}  |  Predicted: {y_pred_real[i]:.2f}")

if __name__ == '__main__':
    main(parse_args())
