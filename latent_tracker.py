import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import comfy.sample
import comfy.samplers
import comfy.model_management
from sklearn.linear_model import LinearRegression

class LatentTracker:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent": ("LATENT",),
            "vae": ("VAE",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
            "overlay_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
        }}

    RETURN_TYPES = ("IMAGE", "LATENT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("heatmap", "latent", "decoded_image", "overlay_image")
    FUNCTION = "track_latent"
    CATEGORY = "sampling/tracking"

    def calculate_linearity(self, trajectory):
        """
        軌跡の直線性（R²値）をGPU上で計算
        trajectory: shape [steps, channels] (GPU tensor)
        """
        steps, channels = trajectory.shape
        x = torch.arange(steps, device=trajectory.device).float()
        x = (x - x.mean()) / x.std()  # 標準化
        x = x.view(-1, 1)  # [steps, 1]

        # 各チャネルの直線性を計算
        r2_scores = []
        for c in range(channels):
            y = trajectory[:, c]
            y = (y - y.mean()) / y.std()  # 標準化

            # 線形回帰の係数を直接計算
            beta = (x.t() @ y) / (x.t() @ x)
            y_pred = x * beta

            # R²値を計算
            ss_tot = torch.sum((y - y.mean()) ** 2)
            ss_res = torch.sum((y - y_pred.squeeze()) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            r2_scores.append(r2)

        # 全チャネルの平均R²値を返す
        return torch.stack(r2_scores).mean()

    def track_latent(self, model, positive, negative, latent, vae, seed, steps, cfg, sampler_name, scheduler, overlay_strength):
        # 全ステップのlatent状態を保存する配列
        latent_states = []
        device = model.model.device
        
        # サンプラーの設定
        noise = comfy.sample.prepare_noise(latent["samples"], seed)
        sigmas = comfy.samplers.calculate_sigmas(model.model.model_sampling, scheduler, steps).cpu()
        
        # プログレスバーのコールバック
        pbar = comfy.utils.ProgressBar(steps)
        def callback(step, denoised, x_t, total_steps):
            latent_states.append(x_t.detach().cpu().numpy())
            pbar.update_absolute(step + 1, total_steps)
            return True

        # サンプリング実行
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent["samples"],
                                    denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                                    force_full_denoise=True, noise_mask=None, sigmas=sigmas, callback=callback, 
                                    disable_pbar=False)  # プログレスバーを有効化

        # latent状態の配列をnumpy配列に変換
        latent_states = np.stack(latent_states)  # [steps, batch, channels, height, width]
        
        # latentの形状を取得
        n_steps, batch_size, n_channels, height, width = latent_states.shape
        
        # モデルをVRAMからアンロード
        model.model.to('cpu')
        torch.cuda.empty_cache()

        # 各点の軌跡の直線性を計算（進捗表示付き）
        print("Calculating linearity for each point...")
        linearity_map = torch.zeros((height, width), device=device)
        
        # latent statesをGPUに転送
        latent_states_gpu = torch.tensor(latent_states, device=device)  # [steps, batch, channels, height, width]

        # バッチ処理で計算を高速化
        batch_size = 32  # バッチサイズを調整可能
        for y in tqdm.tqdm(range(height), desc="Processing rows"):
            for x_start in range(0, width, batch_size):
                x_end = min(x_start + batch_size, width)
                # バッチで複数のxを同時に処理
                trajectories = latent_states_gpu[:, 0, :, y, x_start:x_end]  # [steps, channels, batch]
                trajectories = trajectories.permute(2, 0, 1)  # [batch, steps, channels]
                
                # バッチ内の各点について直線性を計算
                for i, x in enumerate(range(x_start, x_end)):
                    linearity_map[y, x] = self.calculate_linearity(trajectories[i])

        # モデルを再ロード
        model.model.to(device)

        print(f"Mean linearity: {linearity_map.mean().item():.3f}")

        # ヒトマップをプロット
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        
        im = ax.imshow(linearity_map.cpu().numpy(), cmap='viridis', interpolation='nearest')
        ax.set_title(f'Latent Trajectory Linearity (R² values)\nMean: {linearity_map.mean().item():.3f}')
        
        plt.colorbar(im, ax=ax, label='R² value')
        
        # プロットを画像として保存
        fig = plt.gcf()
        fig.canvas.draw()
        plot_image = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close()
        
        # ヒートマップをtensorに変換
        plot_tensor = torch.from_numpy(plot_image[:,:,:3]).float() / 255.0
        plot_tensor = plot_tensor.unsqueeze(0)

        # VAEでlatentをデコード
        decoded = vae.decode(samples)
        # decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
        print(f"Decoded shape: {decoded.shape}")  # [1, 864, 1536, 3]

        # デコード画像はそのまま保持
        # original_decoded = decoded.clone()

        # ヒートマップをデコードされた画像のサイズにリサイズ
        heatmap = linearity_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        target_size = decoded.shape[1:3]  # [864, 1536]
        heatmap = torch.nn.functional.interpolate(heatmap, size=target_size, mode='bicubic')
        heatmap = heatmap.squeeze()  # [H, W]

        # カラーマップを適用
        cm = plt.get_cmap('viridis')
        heatmap_colored = cm(heatmap.cpu().numpy())  # [H, W, 4]
        heatmap_colored = torch.tensor(heatmap_colored[:, :, :3])  # [H, W, 3]
        # NHWC形式に変換
        heatmap_colored = heatmap_colored.unsqueeze(0)  # [1, H, W, 3]
        print(f"Heatmap colored shape: {heatmap_colored.shape}")

        # デバイスを合わせる
        heatmap_colored = heatmap_colored.to(decoded.device)

        # オーバーレイ画像の作成
        overlay = decoded * (1 - overlay_strength) + (heatmap_colored * overlay_strength)
        overlay = torch.clamp(overlay, 0.0, 1.0)

        return (plot_tensor, {"samples": samples}, decoded, overlay) 