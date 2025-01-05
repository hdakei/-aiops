در این پاسخ، کدهای ارائه‌شده را بخش‌بندی کرده و هر قسمت را به‌صورت گام‌به‌گام توضیح می‌دهیم تا سازوکار آن‌ها روشن شود. کد عملاً دو بخش کلی دارد:
	1.	بخش اول: دانلود داده‌های BTC-USD از یاهو فاینانس (Yahoo Finance) با کتابخانهٔ yfinance، اعمال تغییرات اولیه روی ستون‌ها، و در نهایت ذخیرهٔ دیتافریم به فایل CSV.
	2.	بخش دوم: بارگذاری دیتای ذخیره‌شده، ساخت ویژگی‌ها (Feature Engineering)، تعریف چند مدل سادهٔ PyTorch (الهام‌گرفته از SAC, TD3, PPO) و در نهایت ادغام آن‌ها در یک مدل Ensemble برای پیش‌بینی قیمت بیت‌کوین.

بخش اول: دانلود و آماده‌سازی داده با yfinance

کتابخانه‌ها و توابع اصلی

import yfinance as yf
import pandas as pd
import os

	•	yfinance: کتابخانه‌ای برای دریافت داده‌های مالی (سهام، ارز دیجیتال و غیره) از یاهو فاینانس است.
	•	pandas: برای کار با دیتافریم (DataFrame) و پردازش داده.
	•	os: برای کارهای مربوط به فایل و پوشه (چک کردن وجود دایرکتوری، ساخت دایرکتوری، و غیره).

تابع flatten_columns(df)

def flatten_columns(df):
    """
    If the dataframe has multi-index columns (tuples), flatten them into single-level strings.
    """
    flattened = []
    for col in df.columns:
        if isinstance(col, tuple):
            flattened_col = "_".join(map(str, col)).strip()
            flattened.append(flattened_col)
        else:
            flattened.append(col)
    df.columns = flattened
    return df

کاربرد
برخی مواقع کتابخانه‌هایی مثل yfinance دیتافریم‌هایی را برمی‌گردانند که ستون‌هایشان به‌صورت چندسطحی (MultiIndex) هستند. این تابع، اگر ستون‌ها از نوع تاپل (مانند ("Close", "BTC-USD")) باشند، آن‌ها را با اتصال به هم (مثلاً با _) به یک رشته ساده تبدیل می‌کند. در نهایت ستون‌هایی تک‌سطحی (single-level) در دیتافریم خواهیم داشت.

نکات
	•	با استفاده از دستور isinstance(col, tuple) تشخیص می‌دهد که آیا ستون از نوع تاپل است یا خیر.
	•	در صورت تاپل بودن، اعضای آن با _ به هم متصل و تبدیل به استرینگ می‌شوند.
	•	در نهایت دیتافریم را با ستون‌های جدید برمی‌گرداند.

تابع rename_columns_to_lowercase(df)

def rename_columns_to_lowercase(df):
    """
    Convert column names to lowercase, remove spaces/underscores, etc.
    Also remove the '_btc-usd' suffix if present.
    """
    rename_map = {}
    for col in df.columns:
        # Lowercase, replace spaces with underscores
        col_lower = col.lower().replace(" ", "_").strip("_")
        
        # Remove the '_btc-usd' suffix if it exists
        parts = col_lower.split("_")
        if parts[-1] == "btc-usd":
            parts = parts[:-1]  # remove the last part
        col_cleaned = "_".join(parts)
        
        rename_map[col] = col_cleaned
    
    return df.rename(columns=rename_map)

کاربرد
این تابع نام ستون‌ها را ویرایش می‌کند تا:
	1.	همهٔ حروف را به حروف کوچک (Lowercase) تبدیل کند.
	2.	فاصله‌ها را با underscore (_) جایگزین کند.
	3.	اگر پسوند _btc-usd وجود داشته باشد، آن را حذف کند (برای این‌که مثلاً ستون‌هایی مثل Close_BTC-USD به سادگی تبدیل به close شوند).

مراحل
	•	ابتدا با col.lower() نام ستون را کوچک می‌کند.
	•	با col.replace(" ", "_") فاصله‌ها را با _ جایگزین می‌کند.
	•	سپس اگر آخرین بخش رشته همان btc-usd باشد، آن را حذف می‌کند.
	•	در نهایت دیتافریم را با df.rename(columns=rename_map) برمی‌گرداند تا ستون‌ها مطابق rename_map جایگزین شوند.

تابع download_and_debug_btc_data(start_date='2020-01-01', end_date='2023-12-31')

def download_and_debug_btc_data(start_date='2020-01-01', end_date='2023-12-31'):
    """
    1. Downloads BTC-USD data from Yahoo Finance between the specified dates.
    2. Prints columns after download.
    3. Flattens columns (if multi-index).
    4. Prints columns again.
    5. Renames columns to lowercase.
    6. Prints columns again.
    7. Checks for needed columns: open, high, low, close, volume.
    8. Returns a final DataFrame with just the needed columns.
    """
    ticker = 'BTC-USD'
    
    print("Downloading data...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    print("\n** Raw columns from yfinance:")
    print(df.columns)
    
    print("\nFlattening columns (if needed)...")
    df = flatten_columns(df)
    
    print("\n** Columns after flattening:")
    print(df.columns)
    
    print("\nRenaming columns to lowercase...")
    df = rename_columns_to_lowercase(df)
    
    print("\n** Columns after renaming:")
    print(df.columns)

    # Check that we have the needed columns
    needed_cols = ["open", "high", "low", "close", "volume"]
    for col in needed_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' in downloaded DataFrame.")
    
    # Keep only the needed columns
    df = df[needed_cols]
    
    return df

توضیح عملکرد
	1.	yf.download(ticker, start, end): داده‌های BTC-USD را در بازهٔ تاریخی داده‌شده دریافت می‌کند.
	2.	چاپ نام ستون‌های خام (بلافاصله پس از دانلود).
	3.	فراخوانی flatten_columns(df) برای تبدیل ستون‌های چندسطحی به تک‌سطحی.
	4.	دوباره چاپ نام ستون‌ها برای مشاهده تغییرات.
	5.	فراخوانی rename_columns_to_lowercase(df) برای اعمال تغییرات روی نام ستون‌ها و بررسی دوباره.
	6.	بررسی این‌که ستون‌های ضروری (open, high, low, close, volume) وجود داشته باشند.
	7.	دیتافریم نهایی را فقط با ستون‌های مورد نیاز برمی‌گرداند.

تابع main()

def main():
    """
    Main function to download the data, print debug info, and save to CSV if columns are valid.
    """
    start_date = "2020-01-01"
    end_date   = "2023-12-31"
    
    # 1. Download data and debug
    df = download_and_debug_btc_data(start_date, end_date)
    
    # 2. Print sample data
    print("\n** Final DataFrame Head:")
    print(df.head())
    
    # 3. Save to CSV
    output_dir = "./sample_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "BTC_USD_downloaded_debug.csv")
    df.to_csv(output_path)
    print(f"\nData saved to: {output_path}")

if __name__ == "__main__":
    main()

توضیح
	•	این تابع اصلی (Entry point) است.
	•	تاریخ شروع و پایان را تعیین می‌کند.
	•	دیتافریم دانلودشده را می‌گیرد و پس از پردازش، چند سطر اول آن را نشان می‌دهد.
	•	در نهایت آن را در مسیری مشخص (./sample_data/BTC_USD_downloaded_debug.csv) ذخیره می‌کند.
	•	اگر پوشهٔ sample_data وجود نداشته باشد، با دستور os.makedirs(..., exist_ok=True) ساخته می‌شود.

بخش دوم: ساخت مدل Ensemble با PyTorch

در این بخش ابتدا دیتای CSV ذخیره‌شده را بارگذاری می‌کنیم، ویژگی‌های جدید (lags و میانگین متحرک) می‌سازیم و سپس یک مدل PyTorch با سه بلاک مختلف (SAC, TD3, PPO) تعریف می‌کنیم. این سه بلاک خروجی‌هایشان با هم ترکیب شده و در نهایت یک لایه Dense اضافی برای پیش‌بینی داریم.

ایمپورت کتابخانه‌ها

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os

	•	torch, torch.nn, torch.optim: اجزای اصلی کتابخانهٔ PyTorch برای ساخت مدل‌ها و آموزش آن‌ها.
	•	TensorDataset, DataLoader, random_split: برای ساخت دیتاست و دیتالودر و تقسیم به دادهٔ آموزشی و اعتبارسنجی (train/val).
	•	matplotlib.pyplot: برای رسم نمودار.

1. مدل‌ها (SAC, TD3, PPO)

هر سه کلاس تقریباً ساختار یکسانی دارند: یک شبکهٔ تمام‌متصل (Fully Connected) ساده با دو لایه پنهان (Hidden Layer) و خروجی نهایی یک نورون.

کلاس SAC

class SAC(nn.Module):
    """
    Simplified feedforward network inspired by Soft Actor-Critic (SAC).
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(SAC, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)

	•	input_dim: تعداد ویژگی‌های ورودی.
	•	hidden_dim: تعداد نورون‌های هر لایه پنهان.
	•	لایه‌ها:
	•	nn.Linear(input_dim, hidden_dim) + nn.ReLU()
	•	nn.Linear(hidden_dim, hidden_dim) + nn.ReLU()
	•	خروجی: nn.Linear(hidden_dim, 1) (یک مقدار پیش‌بینی می‌کند، مثلاً قیمت).

کلاس TD3

class TD3(nn.Module):
    """
    Simplified feedforward network inspired by Twin Delayed Deep Deterministic (TD3).
    """
    ...

	•	دقیقاً ساختارش شبیه SAC است؛ فقط اسم کلاس متفاوت است تا نشان دهد که این بلاک از ایدهٔ TD3 الهام گرفته.

کلاس PPO

class PPO(nn.Module):
    """
    Simplified feedforward network inspired by Proximal Policy Optimization (PPO).
    """
    ...

	•	باز هم ساختاری مشابه با دو شبکهٔ دیگر دارد.

کلاس EnsembleTradingModel

class EnsembleTradingModel(nn.Module):
    """
    Combines outputs of three RL-based networks (SAC, TD3, PPO) into a single dense layer
    for final price prediction.
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(EnsembleTradingModel, self).__init__()
        self.sac = SAC(input_dim, hidden_dim)
        self.td3 = TD3(input_dim, hidden_dim)
        self.ppo = PPO(input_dim, hidden_dim)
        
        # The final network takes the concatenated outputs of sac, td3, ppo
        self.dense = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # Get each model's output
        sac_out = self.sac(x)
        td3_out = self.td3(x)
        ppo_out = self.ppo(x)
        
        # Concatenate them along dimension=1
        combined = torch.cat([sac_out, td3_out, ppo_out], dim=1)
        
        # Pass through the final dense layer
        out = self.dense(combined)
        return out

توضیح
	•	این کلاس سه مدل جداگانهٔ SAC, TD3, PPO را به‌طور موازی نگه می‌دارد.
	•	در مرحلهٔ forward، ورودی x را به‌صورت جداگانه به هر مدل می‌دهیم و خروجی آن‌ها را در کنار هم قرار می‌دهیم (torch.cat([...], dim=1))، بنابراین در اینجا 3 خروجی عددی داریم که کنار هم می‌آیند.
	•	در نهایت این سه مقدار وارد یک شبکهٔ کوچک (لایهٔ Dense) می‌شوند تا خروجی نهایی (یک اسکالر) ساخته شود.

2. پیش‌پردازش داده‌ها

تابع load_and_preprocess_data(csv_path)

def load_and_preprocess_data(csv_path):
    """
    Reads BTC/USDT CSV data, constructs lag features and a moving average,
    and returns feature tensors and target tensors.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
        
    # Reading Data
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    
    # Feature Engineering:
    # 1) Lags from 1 to 12 on 'close' column
    # 2) MA_10 on 'close'
    # 3) We'll use these as features, and 'close' as the target.
    
    # Ensure 'close' column exists
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")
    
    # Create lag features
    for lag in range(1, 13):
        df[f'lag_{lag}'] = df['close'].shift(lag)
    
    # Moving average (MA_10)
    df['MA_10'] = df['close'].rolling(window=10).mean()
    
    # Drop rows with NaN (early rows from lag/rolling)
    df.dropna(inplace=True)
    
    # Prepare the feature set and target
    feature_cols = [col for col in df.columns if col.startswith('lag_')] + ['MA_10']
    X = df[feature_cols].values
    y = df['close'].values
    
    # Convert to Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    return X_tensor, y_tensor, df

مراحل
	1.	چک می‌شود که فایل CSV وجود دارد یا خیر.
	2.	دیتافریم را از فایل می‌خوانیم، بر اساس تاریخ سورت می‌کنیم.
	3.	ساخت ویژگی‌ها:
	•	lag_1 تا lag_12 از ستون close. یعنی در هر سطر، مقدارهای n روز قبلِ قیمت را به‌عنوان ویژگی استفاده می‌کنیم.
	•	محاسبهٔ میانگین متحرک 10 روزه (MA_10).
	4.	با توجه به این‌که lag و rolling باعث می‌شود سطرهای ابتدایی مقادیر تهی داشته باشند، آن‌ها را حذف می‌کنیم (df.dropna).
	5.	تعیین اینکه چه ستون‌هایی ویژگی (X) باشند؛ در اینجا تمام lag_ها + MA_10.
	6.	ستون close به‌عنوان هدف (y) در نظر گرفته می‌شود.
	7.	تبدیل آرایه‌های numpy به torch.tensor.
	8.	دیتافریم نهایی هم برگردانده می‌شود (ممکن است برای آنالیزهای بعدی مفید باشد).

تابع create_dataloaders(X_tensor, y_tensor, train_ratio=0.8, batch_size=32)

def create_dataloaders(X_tensor, y_tensor, train_ratio=0.8, batch_size=32):
    """
    Splits the dataset into training and validation sets, returns corresponding DataLoaders.
    """
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

توضیح
	•	ابتدا یک TensorDataset از X و y می‌سازد.
	•	سپس با توجه به train_ratio (در اینجا 0.8 یا 80%)، دیتاست را به دو قسمت train/val تقسیم می‌کند.
	•	در نهایت دو DataLoader می‌سازد: یکی برای Train (با shuffle=True) و یکی برای Validation (با shuffle=False).

3. آموزش مدل

تابع train_ensemble_model(model, train_loader, val_loader, epochs=50, lr=1e-3)

def train_ensemble_model(model, train_loader, val_loader, epochs=50, lr=1e-3):
    """
    Trains the EnsembleTradingModel using MSE loss and Adam optimizer.
    Returns training and validation losses for each epoch.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        batch_train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
        
        # Compute average training loss
        avg_train_loss = np.mean(batch_train_losses)
        
        # Validation phase
        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                batch_val_losses.append(loss.item())
        avg_val_loss = np.mean(batch_val_losses)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    
    return train_losses, val_losses

مراحل آموزش
	1.	تعیین معیار خطا (nn.MSELoss) و بهینه‌ساز (Adam) با نرخ یادگیری lr.
	2.	یک حلقه روی اپوک‌ها (epochs) می‌زنیم.
	3.	در هر اپوک:
	•	مد را روی model.train() قرار می‌دهیم تا لایه‌های Dropout یا BatchNorm (در صورت وجود) رفتاری مطابق آموزش داشته باشند (اگرچه در این شبکه ساده نیازی نیست).
	•	برای هر بچ از train_loader خروجی مدل را حساب می‌کنیم، خطا را به‌دست می‌آوریم، و سپس loss.backward() برای محاسبه‌ی گرادیان. در انتها optimizer.step() برای آپدیت وزن‌ها.
	•	میانگین خطای Train ثبت می‌شود.
	4.	سپس برای فاز اعتبارسنجی (Validation) مدل را در حالت model.eval() قرار می‌دهیم و بدون اعمال backward() خطای اعتبارسنجی را حساب می‌کنیم.
	5.	در نهایت لیست خطاهای train و val را به‌عنوان خروجی برمی‌گرداند.

4. ارزیابی و مصورسازی

تابع evaluate_model(model, X_tensor, y_tensor)

def evaluate_model(model, X_tensor, y_tensor):
    """
    Obtains model predictions on the entire dataset, calculates MSE, MAE.
    Returns predictions and metrics (mse, mae).
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).view(-1)
    
    y_true = y_tensor.view(-1)
    mse = nn.MSELoss()(y_pred, y_true).item()
    mae = nn.L1Loss()(y_pred, y_true).item()
    
    return y_pred.detach().numpy(), mse, mae

	•	مدل را در حالت eval می‌برد و بدون محاسبه‌ی گرادیان خروجی را حساب می‌کند.
	•	معیارهای MSE و MAE روی کل داده حساب می‌شوند و در نهایت برمی‌گرداند.

تابع plot_losses(train_losses, val_losses)

def plot_losses(train_losses, val_losses):
    """
    Plots training and validation loss curves over epochs.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.show()

	•	نمودار خطاهای آموزشی و اعتبارسنجی را در طول اپوک رسم می‌کند.

تابع plot_predictions(y_true, y_pred, title="Model Predictions vs. Actuals")

def plot_predictions(y_true, y_pred, title="Model Predictions vs. Actuals"):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.show()

	•	نمودار سری زمانی قیمت واقعی (y_true) در مقابل قیمت پیش‌بینی‌شده (y_pred) را رسم می‌کند.

5. تابع main() در بخش دوم

def main():
    # Define the path to your preprocessed CSV file
    csv_path = "./sample_data/BTC_USD_downloaded_debug.csv"
    
    # 1. Load and preprocess data
    X_tensor, y_tensor, df = load_and_preprocess_data(csv_path)
    
    # 2. Create train/validation DataLoaders
    train_loader, val_loader = create_dataloaders(X_tensor, y_tensor, train_ratio=0.8, batch_size=32)
    
    # 3. Initialize the Ensemble Model
    input_dim = X_tensor.shape[1]  # number of features
    model = EnsembleTradingModel(input_dim, hidden_dim=64)
    
    # 4. Train the model
    train_losses, val_losses = train_ensemble_model(model, train_loader, val_loader, epochs=50, lr=1e-3)
    
    # 5. Evaluate the model
    y_pred, mse, mae = evaluate_model(model, X_tensor, y_tensor)
    
    # 6. Visualization
    #    a) Loss plots
    plot_losses(train_losses, val_losses)
    
    #    b) Predictions vs. Actual
    plot_predictions(y_tensor.numpy(), y_pred, title="BTC-USDT Closing Price: Ensemble Model")
    
    #    c) Print error metrics
    print(f"Final MSE: {mse:.6f}")
    print(f"Final MAE: {mae:.6f}")

if __name__ == "__main__":
    main()

جریان کار
	1.	مسیر فایل CSV را که در بخش اول ساخته‌ایم، تعیین می‌کند.
	2.	داده‌ها را بارگذاری و پیش‌پردازش می‌کند (ساخت Lag و MA).
	3.	دیتالودرهای آموزشی/اعتبارسنجی می‌سازد.
	4.	مدل Ensemble (ترکیبی از SAC, TD3, PPO) را با ورودی در ابعاد تعداد ویژگی‌های ما (input_dim) و تعداد نورون‌های پنهان (دلخواه) ایجاد می‌کند.
	5.	مدل را با استفاده از تابع train_ensemble_model به مدت 50 اپوک آموزش می‌دهد.
	6.	مدل را روی کل داده‌ها ارزیابی می‌کند و خروجی (MSE و MAE) را می‌گیرد.
	7.	نتایج را به‌صورت نمودار نمایش می‌دهد (نمودار خطا و نمودار پیش‌بینی در برابر واقعی).

جمع‌بندی
	1.	بخش اول کد، با yfinance دیتای BTC-USD را دانلود کرده و با flatten/rename ستون‌ها، دیتافریم نهایی را در قالب CSV ذخیره می‌کند.
	2.	بخش دوم کد، CSV حاصل را بارگذاری، ویژگی‌های مختلف ایجاد، و با یک مدل Ensemble بر اساس سه شبکهٔ سادهٔ PyTorch (SAC, TD3, PPO) قیمت آیندهٔ بیت‌کوین را پیش‌بینی می‌کند.
	3.	در نهایت، از طریق نمایش خطاها (MSE, MAE) و نمودارهای مقایسهٔ پیش‌بینی و مقدار واقعی، کیفیت مدل سنجیده می‌شود.

با اجرای این دو فایل پشت سر هم، ابتدا دیتای خام را گرفته و پاک‌سازی می‌کنیم (بخش اول)، سپس مدل PyTorch را روی داده آموزش داده و نتایج را مشاهده می‌کنیم (بخش دوم).
