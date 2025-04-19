import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

def initialize_ui():
    # 創建主視窗
    root = tk.Tk()
    root.title("Hopfield Network Visualization")
    root.geometry("1200x500")

    # 創建頂部框架（包含下拉選單和按鈕）
    frame_top = ttk.Frame(root)
    frame_top.pack(pady=10)

    # 創建下拉選單
    global dataset_var  # 設為全域變數
    dataset_var = tk.StringVar()
    dataset_combo = ttk.Combobox(frame_top, textvariable=dataset_var, values=["Basic", "Bonus"])
    dataset_combo.set("Basic")
    dataset_combo.pack(side=tk.LEFT, padx=5)

    # 創建Previous按鈕
    global previous_button  # 設為全域變數
    previous_button = ttk.Button(frame_top, text="Previous", command=previous_pattern)
    previous_button.pack(side=tk.LEFT, padx=5)

    # 創建Next按鈕
    global next_button  # 設為全域變數
    next_button = ttk.Button(frame_top, text="Next", command=next_pattern)
    next_button.pack(side=tk.LEFT, padx=5)

    # 創建Test Noise按鈕
    noise_button = ttk.Button(frame_top, text="Test Noise", command=test_noise)
    noise_button.pack(side=tk.LEFT, padx=5)

    # 創建圖形顯示區域的框架
    global frame_plots  # 設為全域變數
    frame_plots = ttk.Frame(root)
    frame_plots.pack(pady=10, expand=True, fill=tk.BOTH)

    # 初始化全域變數
    global current_index, train_patterns, test_patterns, rows, cols, W, theta
    current_index = 0
    train_patterns = []
    test_patterns = []
    rows, cols = 9, 12
    W = None
    theta = None

    # 綁定下拉選單的事件
    dataset_combo.bind('<<ComboboxSelected>>', update_display)

    # 初始顯示
    update_display()

    return root

def convert_to_array(file_path): # 將txt檔案轉換為2D陣列
    with open(file_path, 'r') as file:
        data = file.read().split('\n\n')  # 分割每個圖形
        result = []
        for pattern in data:
            pattern_array = []
            for line in pattern.split('\n'):
                line_array = [-1 if char == ' ' else 1 for char in line]
                pattern_array.append(line_array)
            result.append(pattern_array)
        return result

def train_hopfield(patterns): # 訓練Hopfield網路
    # 將每個圖案轉換為向量
    vectors = []
    for pattern in patterns:
        # 將2D陣列展平為1D向量
        vector = np.array(pattern).flatten()
        vectors.append(vector)
    
    vectors = np.array(vectors)
    p = vectors.shape[1]  # 向量維度
    N = len(vectors)      # 訓練樣本數

    
    # 計算權重矩陣
    W = np.zeros((p, p))
    for x in vectors:
        W += np.outer(x, x)
    W = W/p - (N/p) * np.eye(p)
    
    # 計算閾值
    # theta = np.sum(W, axis=1)
    theta = np.zeros(p)  # 考慮不使用閾值
    
    return W, theta

def recall_hopfield(input_pattern, W, theta, max_iterations=100): # 回想Hopfield網路
    # 將輸入圖案轉換為向量
    x = np.array(input_pattern).flatten()
    
    for iter_count in range(max_iterations):
        x_old = x.copy()
        
        # 更新每個神經元
        # print("theta : ", theta.shape)
        net = np.dot(W, x) - theta
        # print("net : ", net.shape)
        x = np.sign(net)
        # print("x : ", x.shape)
        
    #     # 檢查是否收斂
    #     if np.array_equal(x, x_old):
    #         print(f"Converged after {iter_count + 1} iterations ------------------------")
    #         break
    # else:
    #     print(f"Did not converge after {max_iterations} iterations ------------------------")
    
    # 將結果轉回原始形狀
    rows = len(input_pattern)
    cols = len(input_pattern[0])
    result = x.reshape(rows, cols)
    
    return result

def update_display(*args): # 更新顯示
    global current_index, train_patterns, test_patterns, rows, cols, W, theta
    # 清除現有的圖形
    for widget in frame_plots.winfo_children():
        widget.destroy()
    
    # 根據選擇讀取對應的檔案
    selected = dataset_var.get()
    if selected == "Basic":
        train_file = 'Hopfield_dataset/Basic_Training.txt'
        test_file = 'Hopfield_dataset/Basic_Testing.txt'
        rows, cols = 9, 12
    else:  # Bonus
        train_file = 'Hopfield_dataset/Bonus_Training.txt'
        test_file = 'Hopfield_dataset/Bonus_Testing.txt'
        rows, cols = 10, 10
    
    # 讀取資料
    train_patterns = convert_to_array(train_file)
    test_patterns = convert_to_array(test_file)
    
    # 訓練 Hopfield 網路
    W, theta = train_hopfield(train_patterns)
    
    current_index = 0  # 重置索引
    display_patterns()
    update_button_states()

def display_patterns(): # 顯示圖案
    # 對當前的測試樣本進行回想
    result = recall_hopfield(test_patterns[current_index], W, theta)
    
    # 創建三個子圖（訓練資料、測試資料、結果）
    titles = ['Training Pattern', 'Testing Pattern', 'Result']
    patterns = [train_patterns[current_index], test_patterns[current_index], result]
    
    for i in range(3):
        fig = plt.Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=frame_plots)
        canvas.get_tk_widget().pack(side=tk.LEFT, padx=5)
        
        # 繪製網格圖
        ax.imshow(patterns[i], cmap='binary')
        ax.grid(True, which='minor', color='black', linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 根據資料集類型設置網格
        if dataset_var.get() == "Basic":
            ax.set_xticks(np.arange(-0.5, 9, 1), minor=True)   # x 軸 9 格
            ax.set_yticks(np.arange(-0.5, 12, 1), minor=True)  # y 軸 12 格
        else:  # Bonus
            ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        
        # 設置標題
        if i == 0:  # 只為訓練模式添加編號
            ax.set_title(f'Pattern {current_index + 1}\n{titles[i]}', 
                        loc='left',
                        pad=10,
                        y=0.97)
        else:
            ax.set_title(titles[i])

def update_button_states(): # 更新按鈕狀態
    if current_index == 0:
        previous_button.state(['disabled'])  # 禁用 Previous 按鈕
    else:
        previous_button.state(['!disabled'])  # 啟用 Previous 按鈕
        
    if current_index == len(train_patterns) - 1:
        next_button.state(['disabled'])  # 禁用 Next 按鈕
    else:
        next_button.state(['!disabled'])  # 啟用 Next 按鈕

def next_pattern(): # 下一個圖案
    global current_index
    if current_index < len(train_patterns) - 1:
        current_index += 1
        # 清除現有的圖形
        for widget in frame_plots.winfo_children():
            widget.destroy()
        display_patterns()
        update_button_states()
        # 印出當前資料集和pattern編號
        dataset_type = dataset_var.get()
        print(f"{dataset_type} data Pattern {current_index + 1} displayed.")
        print()

def previous_pattern(): # 上一個圖案
    global current_index
    if current_index > 0:
        current_index -= 1
        # 清除現有的圖形
        for widget in frame_plots.winfo_children():
            widget.destroy()
        display_patterns()
        update_button_states()
        # 印出當前資料集和pattern編號
        dataset_type = dataset_var.get()
        print(f"{dataset_type} data Pattern {current_index + 1} displayed.")
        print()

def add_noise(pattern, noise_rate=0.25): # 將圖案加入雜訊
    noisy_pattern = np.array(pattern) # 將圖案轉換為numpy陣列   
    total_pixels = noisy_pattern.size # 計算圖案的總像素數
    num_pixels_to_flip = int(total_pixels * noise_rate) # 計算要翻轉的像素數
    
    # 隨機選擇要翻轉的像素
    indices = np.random.choice(total_pixels, num_pixels_to_flip, replace=False) # 隨機選擇要翻轉的像素
    flat_pattern = noisy_pattern.flatten() # 將圖案展平為1D向量
    flat_pattern[indices] *= -1  # 翻轉選中的像素
    
    return flat_pattern.reshape(noisy_pattern.shape) # 將1D向量轉換回2D陣列

def test_noise(): # 測試雜訊
    global current_index
    # 對當前圖案加入雜訊
    noisy_pattern = add_noise(train_patterns[current_index])
    
    # 使用雜訊圖案進行回想
    result = recall_hopfield(noisy_pattern, W, theta)
    
    # 清除現有的圖形
    for widget in frame_plots.winfo_children():
        widget.destroy()
    
    # 顯示原始、雜訊和回想結果
    titles = ['Original Pattern', 'Noisy Pattern', 'Recalled Result']
    patterns = [train_patterns[current_index], noisy_pattern, result]
    
    for i in range(3):
        fig = plt.Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=frame_plots)
        canvas.get_tk_widget().pack(side=tk.LEFT, padx=5)
        
        # 繪製網格圖
        ax.imshow(patterns[i], cmap='binary')
        ax.grid(True, which='minor', color='black', linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 根據資料集類型設置網格
        if dataset_var.get() == "Basic":
            ax.set_xticks(np.arange(-0.5, 9, 1), minor=True)   # x 軸 9 格
            ax.set_yticks(np.arange(-0.5, 12, 1), minor=True)  # y 軸 12 格
        else:  # Bonus
            ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        
        # 設置標題
        if i == 0:  # 只為原始模式添加編號
            ax.set_title(f'Pattern {current_index + 1}\n{titles[i]}', 
                        loc='left',
                        pad=10,
                        y=0.97)
        else:
            ax.set_title(titles[i])
    
    # 印出當前資料集和pattern編號
    dataset_type = dataset_var.get()
    print(f"{dataset_type} data Pattern {current_index + 1} noise test completed!")
    print()


if __name__ == "__main__":
    root = initialize_ui()  # 只需要返回 root
    # 啟動主循環
    root.mainloop()
