import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FixedLocator


def plot_delay(delays_runs, true_delays, xdima, ydims, fig_dir, file_name, binsize=1):
    T = delays_runs.shape[1]
    num_pairs = int(len(ydims)*(len(ydims)-1)/2)
    delays_runs = np.concatenate((np.zeros((delays_runs.shape[0], delays_runs.shape[1], 1, xdima)), delays_runs), axis=-2)
    delays_pairwise = np.zeros(((delays_runs.shape[0], num_pairs, T, xdima)))
    num_delay_dim = delays_runs.shape[2]

    if true_delays is not None:
        true_delays = np.concatenate((np.zeros((true_delays.shape[0], 1, xdima)), true_delays), axis=1)
        true_delays_pairwise = np.zeros(((num_pairs, T, xdima)))

    for xi in range(xdima):
        count = 0

        for i in range(num_delay_dim):
            for j in range(i+1, num_delay_dim):
                delays_pairwise[:, count, :, xi] = delays_runs[:,:,j, xi] - delays_runs[:,:,i, xi]
                if true_delays is not None:
                    true_delays_pairwise[count, :, xi] = true_delays[:, j, xi] - true_delays[:, i, xi]
                count += 1

        avg_delays_pairwise = np.mean(delays_pairwise[..., xi], axis=0) * binsize
        avg_delays_pairwise_std = np.std(delays_pairwise[..., xi], axis=0) * binsize

        begin_idx = 0
        fig = plt.figure(figsize=(4*(int(num_pairs / 2) + 1), 3))
        for i in range(delays_pairwise.shape[1]):
            plt.subplot(2, int(num_pairs / 2) + 1, i+1)
            plt.plot(avg_delays_pairwise[i,begin_idx:], linewidth=0.75, color="darkred", label=f"learned for dim {xi+1}")
            plt.fill_between(np.arange(0,len(avg_delays_pairwise[i, begin_idx:])), avg_delays_pairwise[i, begin_idx:] - avg_delays_pairwise_std[i,begin_idx:], avg_delays_pairwise[i,begin_idx:] + avg_delays_pairwise_std[i,begin_idx:], 
                    color='red', alpha=0.15)

            if true_delays is not None:
                plt.plot(true_delays_pairwise[i, begin_idx:, xi] * binsize, linestyle=(0, (1, 5)), linewidth=0.75, color="purple", label="true delay")
            plt.legend(loc=1)
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            plt.xticks([])

        plt.savefig(f"{fig_dir}delay_{file_name}.png", dpi=300)
        
        # return fig

def plot_latent(pred_xs_runs, true_xs_runs, xdima, xdimw, ydims, fig_dir, file_name):
    # stack all runs:   → (n_run, trials, T, lat)
    pred_xs_runs = np.stack(pred_xs_runs, 0)
    xs_0 = pred_xs_runs[:, 0, :, :]
    if true_xs_runs is not None:
        true_xs_runs = np.stack(true_xs_runs, 0)
        xs_0_true = true_xs_runs[:, 0, :, :]                         # (n_run, T, lat)

    plot_dim   = xdima + xdimw if xdima > 0 else 1
    n_regions  = len(ydims)
    n_runs     = len(pred_xs_runs)              # → args.num_repeat

    n_rows = n_runs * n_regions
    fig_lat, axes_lat = plt.subplots(xdima, n_rows,
                                    figsize=(5*n_rows, 2*xdima),
                                    sharex=True, squeeze=False)
    for r in range(n_runs):
        for reg in range(n_regions):
            row = r * n_regions + reg
            for lat in range(xdima):
                ax = axes_lat[lat, row]
                lat_idx = lat + reg*plot_dim          # latent index in concat vector
                if true_xs_runs is not None:
                    index_1 = np.argmax(np.abs(xs_0[r, :, lat_idx]))
                    index_2 = np.argmax(np.abs(xs_0_true[r, :, lat_idx]))
                    sign = 1 if np.allclose(xs_0[r, index_1, lat_idx] - xs_0_true[r, index_2, lat_idx], 0, atol=1e-1) else -1
                else:
                    sign = 1 
                ax.plot(xs_0[r, :, lat_idx], lw=1.0, label="learned")
                if true_xs_runs is not None:
                    ax.plot(sign * xs_0_true[r, :, lat_idx], lw=1.0, label="true")
                ax.legend(loc=1)
                if r == 0: ax.set_title(f"across-region latent {lat+1}")
                if lat == 0:
                    ax.set_ylabel(f"run {r+1} / region {reg+1}")
                if row == xdima-1: ax.set_xlabel("time-bin")

    fig_lat.suptitle(f"latents of trial 1 "
                    f"({n_runs} runs * {n_regions} regions)")
    fig_lat.tight_layout()
    fig_lat.savefig(f"{fig_dir}latent_{file_name}.png", dpi=300)
    
    return fig_lat

def plot_loss(train_losses_runs, val_losses_runs, avg_test_loss_runs, fig_dir, file_name):
    fig_loss, ax_loss = plt.subplots(1, 2, figsize=(14, 4), sharex=True)

    num_repeat = len(train_losses_runs)

    for k in range(len(train_losses_runs)):
        ax_loss[0].plot(train_losses_runs[k], lw=0.7, alpha=0.6)
        ax_loss[1].plot(val_losses_runs[k], lw=0.7, alpha=0.6)

    for p,mat,ttl in zip(ax_loss, [train_losses_runs, val_losses_runs], ["train","val"]):
        mean, std = np.nanmean(mat,0), np.nanstd(mat,0)
        p.plot(mean, lw=2, color='k', label='mean')
        p.fill_between(np.arange(len(mean)), mean-std, mean+std, alpha=0.25, color="grey")
        p.set_title(f"{ttl} loss"); p.set_xlabel("epoch"); p.set_ylabel("loss")
        p.legend()

    mean_loss = np.mean(avg_test_loss_runs)
    std_loss  = np.std(avg_test_loss_runs)
    fig_loss.suptitle(
        f"loss over {num_repeat} runs, "
        f"test loss {mean_loss:.2e} ± {std_loss:.2e}"
    )
    fig_loss.tight_layout()
    fig_loss.savefig(f"{fig_dir}loss_{file_name}.png", dpi=300)
    return fig_loss

def plot_observation(pred_ys_runs, true_ys_runs, pred_ys_train_record_runs, true_ys_train_record_runs, fig_dir, file_name):
    n_run = len(pred_ys_runs)
    fig_obs, axes = plt.subplots(2*n_run, 2, figsize=(10, 4*n_run), sharex=True)

    test_mse_runs = []
    train_mse_runs = []
    for r in range(n_run):
        test_mse = np.mean((true_ys_runs[r] - pred_ys_runs[r]) ** 2)
        train_mse = np.mean((true_ys_train_record_runs[r] - pred_ys_train_record_runs[r]) ** 2)
        # --- train ---
        tr_true = np.mean(true_ys_train_record_runs[r],0)
        tr_pred = np.mean(pred_ys_train_record_runs[r],0)
        im = axes[2*r,0].imshow(tr_true.T, aspect='auto'); plt.colorbar(im, ax=axes[2*r,0])
        axes[2*r,0].set_title(f"run {r+1}  train TRUE")
        im = axes[2*r,1].imshow(tr_pred.T, aspect='auto'); plt.colorbar(im, ax=axes[2*r,1])
        axes[2*r,1].set_title(f"run {r+1}  train PRED with MSE {train_mse}")
        # --- test ---
        te_true = np.mean(true_ys_runs[r],0)
        te_pred = np.mean(pred_ys_runs[r],0)
        im = axes[2*r+1,0].imshow(te_true.T, aspect='auto'); plt.colorbar(im, ax=axes[2*r+1,0])
        axes[2*r+1,0].set_title(f"run {r+1}  test TRUE")
        im = axes[2*r+1,1].imshow(te_pred.T, aspect='auto'); plt.colorbar(im, ax=axes[2*r+1,1])
        axes[2*r+1,1].set_title(f"run {r+1}  test PRED with MSE {test_mse}")

  
        axes[2*r,0].set_xlabel("time-bin")
        axes[2*r,1].set_xlabel("time-bin")
        axes[2*r+1,0].set_xlabel("time-bin")
        axes[2*r+1,1].set_xlabel("time-bin")

        axes[2*r,0].set_ylabel("neurons")
        axes[2*r,1].set_ylabel("neurons")
        axes[2*r+1,0].set_ylabel("neurons")
        axes[2*r+1,1].set_ylabel("neurons")

        train_mse_runs.append(train_mse)
        test_mse_runs.append(test_mse)

    mean_train_mse = np.mean(train_mse_runs)
    std_train_mse = np.std(train_mse_runs)
    mean_test_mse = np.mean(test_mse_runs)
    std_test_mse = np.std(test_mse_runs)
    fig_obs.suptitle(
        f"observations across {n_run} runs\n"
        f"train MSE {mean_train_mse:.4e} ± {std_train_mse:.4e}\n"
        f"test  MSE {mean_test_mse:.4e} ± {std_test_mse:.4e}"
    )
    fig_obs.savefig(f"{fig_dir}observation_{file_name}.png", dpi=300)
    fig_obs.tight_layout()
    return fig_obs
