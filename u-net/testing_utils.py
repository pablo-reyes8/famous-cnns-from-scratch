import numpy as np, torch, matplotlib.pyplot as plt
from scipy import ndimage as ndi
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.manifold import TSNE
try:
    import umap
    HAVE_UMAP = True
except:
    HAVE_UMAP = False
from scipy.spatial.distance import directed_hausdorff


def unnormalize_img(x_chw, mean, std):
    """
    x_chw: tensor [C,H,W] normalizado
    mean,std: tu normalización por canal (tuplas/listas de len C)
    return: np.float32 en [0,1]
    """
    x = x_chw.detach().cpu().float().clone()
    for c in range(x.size(0)):
        x[c] = x[c] * std[c] + mean[c]
    x = x.clamp(0, 1)
    return x.permute(1,2,0).numpy()


def get_logits(out):
    """Convierte la salida del modelo a un Tensor de logits."""
    if isinstance(out, (tuple, list)):
        return out[0]
    if isinstance(out, dict):
        return out.get('out', next(iter(out.values())))
    return out


def clear_all_hooks(module):
    # borra hooks de este módulo
    if hasattr(module, "_forward_hooks"):       module._forward_hooks.clear()
    if hasattr(module, "_forward_pre_hooks"):   module._forward_pre_hooks.clear()
    if hasattr(module, "_backward_hooks"):      module._backward_hooks.clear()
    # recursivo
    for m in module.children():
        clear_all_hooks(m)



def viz_overlay_errors(xb, out_or_logits, yb, thr=0.5, mean=None, std=None, titles=True):
    """
    Visualiza imagen, mapa de probabilidad y mapa de errores (R=FP, G=TP, B=FN).

    xb:  [1,C,H,W]
    out_or_logits: Tensor logits [1,1,H,W] o salida cruda del modelo (tuple/dict/Tensor)
    yb:  [1,1,H,W] o [1,H,W] con {0,1}
    thr: umbral de probabilidad para binarizar (binario)
    mean/std: tu normalización (p.ej. ImageNet) para desnormalizar la imagen
    """
    assert xb.size(0) == 1, "Pasa batch de tamaño 1 para visualizar"
    x = xb[0]

    # Imagen a [H,W,3] en [0,1]
    if mean is not None and std is not None:
        img = unnormalize_img(x, mean, std)
    else:
        img = x.detach().cpu().permute(1,2,0).float().numpy()
        # si C=1, replicamos canales para mostrar en RGB
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin + 1e-8)
    img = np.clip(img, 0, 1)

    # Logits -> probas
    logits = get_logits(out_or_logits)
    prob = torch.sigmoid(logits)[0,0].detach().cpu().numpy()

    # Ground truth a [H,W] uint8
    y = yb[0]
    if y.dim() == 3:   # [1,H,W] -> [H,W]
        y = y[0]
    gt = y.detach().cpu().numpy().astype(np.uint8)

    # Predicción binaria
    pred = (prob > thr).astype(np.uint8)

    # Contornos (opcional)
    edge_pred = np.logical_xor(pred, ndi.binary_erosion(pred))
    edge_gt   = np.logical_xor(gt,   ndi.binary_erosion(gt))

    # Mapa de errores RGB
    err = np.zeros((*pred.shape, 3), dtype=float)
    err[...,0] = (pred == 1) & (gt == 0)   # FP → rojo
    err[...,1] = (pred == 1) & (gt == 1)   # TP → verde
    err[...,2] = (pred == 0) & (gt == 1)   # FN → azul

    # Plots
    plt.figure(figsize=(14,4))
    ax = plt.subplot(1,3,1); ax.imshow(img); ax.axis('off')
    if titles: ax.set_title("Imagen")

    ax = plt.subplot(1,3,2); ax.imshow(img); ax.imshow(prob, alpha=0.5); ax.axis('off')
    if titles: ax.set_title("Probabilidad (sigmoid)")

    ax = plt.subplot(1,3,3); ax.imshow(img*0.6); ax.imshow(err, alpha=0.6); ax.axis('off')
    if titles: ax.set_title("Errores  (R=FP, G=TP, B=FN)")
    plt.tight_layout(); plt.show()


def plot_pr_roc_from_logits(logits, yb, mask_valid=None):
    """
    logits: [B,1,H,W], yb: [B,1,H,W] o [B,H,W] {0,1}
    mask_valid: opcional [B,H,W] o [B,1,H,W] (True donde evaluar)
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    y = yb.detach().cpu()
    if y.dim()==3: y = y.unsqueeze(1)
    y = y.numpy().reshape(-1)

    if mask_valid is not None:
        m = mask_valid
        if isinstance(m, torch.Tensor): m = m.detach().cpu().numpy()
        m = m.reshape(-1).astype(bool)
        probs, y = probs[m], y[m]

    p, r, _ = precision_recall_curve(y, probs)
    ap = average_precision_score(y, probs)
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(r, p); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AP={ap:.3f})"); plt.grid(True, alpha=.3)
    plt.subplot(1,2,2); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={roc_auc:.3f})"); plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.show()


def unnormalize_img(x_chw: torch.Tensor, mean=None, std=None):
    x = x_chw.detach().cpu().float().clone()
    if mean is not None and std is not None:
        for c in range(x.shape[0]):
            x[c] = x[c] * std[c] + mean[c]
        x = x.clamp(0, 1)
        img = x.permute(1,2,0).numpy()
    else:
        img = x.permute(1,2,0).numpy()
        if img.shape[2] == 1:  # grayscale -> RGB
            img = np.repeat(img, 3, axis=2)
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin + 1e-8)
        img = np.clip(img, 0, 1)
    return img

def mc_dropout_predict(model, xb, T=20, num_classes=1):
    """
    Devuelve media y varianza de probas con MC Dropout.
    num_classes=1 -> usa sigmoid; >=2 -> softmax (devuelve probas completas [B,C,H,W]).
    """
    was_training = model.training
    model.train()  # activar dropout en inferencia
    with torch.no_grad():
        probs_list = []
        for _ in range(int(T)):
            out = model(xb)
            logits = get_logits(out)
            if num_classes == 1:
                p = torch.sigmoid(logits)            # [B,1,H,W]
            else:
                p = torch.softmax(logits, dim=1)     # [B,C,H,W]
            probs_list.append(p)
        probs = torch.stack(probs_list, dim=0)        # [T,B,1,H,W] o [T,B,C,H,W]
        mean  = probs.mean(0)
        var   = probs.var(0)
    if not was_training:
        model.eval()
    return mean, var   # shapes: [B,1,H,W] y [B,1,H,W] (bin) o [B,C,H,W] (mc)

def plot_uncertainty(xb, mean, var, idx=0, mean_std=None):
    """
    mean_std: (mean, std) si quieres desnormalizar la imagen; si no, hace min-max.
    Para multiclase, puedes visualizar una clase específica con mean[:,k] y var[:,k].
    """
    if isinstance(mean_std, tuple) and len(mean_std) == 2:
        mean_img, std_img = mean_std
    else:
        mean_img = std_img = None

    img = unnormalize_img(xb[idx], mean=mean_img, std=std_img)
    # Para binario: canales [1,H,W]; para multiclase puedes escoger canal k
    if mean.ndim == 4:   # [B,1,H,W]
        m = mean[idx,0].detach().cpu().numpy()
        s2 = var[idx,0].detach().cpu().numpy()
    else:
        raise ValueError("mean/var deben ser [B,1,H,W] (binario) o adapta para multiclase seleccionando canal.")

    plt.figure(figsize=(12,4))
    ax = plt.subplot(1,3,1); ax.set_title("Imagen"); ax.imshow(img); ax.axis('off')
    ax = plt.subplot(1,3,2); ax.set_title("Prob. media"); ax.imshow(img); ax.imshow(m, alpha=0.5); ax.axis('off')
    ax = plt.subplot(1,3,3); ax.set_title("Varianza (incertidumbre)"); im = ax.imshow(s2); ax.axis('off'); plt.colorbar(im, fraction=0.046)
    plt.tight_layout(); plt.show()

def dice_per_image_from_logits(logits, yb, thr=0.5, eps=1e-7):
    """
    retorna lista de Dice por imagen
    """
    probs = torch.sigmoid(logits)
    if yb.dim()==3: yb = yb.unsqueeze(1)
    y = yb.float()
    pred = (probs > thr).float()

    inter = (pred*y).sum(dim=(1,2,3))
    sums = pred.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3))
    dice = (2*inter + eps)/(sums + eps)                 # [B]
    return dice.detach().cpu().numpy()

def iou_per_image_from_logits(logits, yb, thr=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    if yb.dim()==3: yb = yb.unsqueeze(1)
    y = yb.float()
    pred = (probs > thr).float()
    inter = (pred*y).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3)) - inter
    iou = (inter + eps)/(union + eps)
    return iou.detach().cpu().numpy()

def plot_hist_metrics(dices, ious):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.hist(dices, bins=20); plt.title(f"Dice (mean={np.mean(dices):.3f})")
    plt.subplot(1,2,2); plt.hist(ious,  bins=20); plt.title(f"IoU  (mean={np.mean(ious):.3f})")
    plt.tight_layout(); plt.show()



def calibration_curve_pixels(logits, yb, n_bins=10):
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    y = yb.detach().cpu()
    if y.dim()==3: y = y.unsqueeze(1)
    y = y.numpy().reshape(-1)

    bins = np.linspace(0,1,n_bins+1)
    binids = np.digitize(probs, bins)-1
    conf, acc = [], []
    for b in range(n_bins):
        m = (binids==b)
        if m.sum()==0:
            conf.append(np.nan); acc.append(np.nan)
        else:
            conf.append(probs[m].mean())
            acc.append(y[m].mean())
    conf, acc = np.array(conf), np.array(acc)
    plt.figure(figsize=(4.5,4.5))
    plt.plot([0,1],[0,1],'--',alpha=.5)
    plt.plot(conf, acc, marker='o')
    plt.xlabel("Confianza promedio (p)"); plt.ylabel("Frecuencia empírica")
    plt.title("Calibración (pixel-level)"); plt.grid(True, alpha=.3); plt.show()


def plot_learning_curves(hist_train, hist_val, metric_key="Dice"):
    """
    hist_*: lista de dicts {'loss':..., 'pix_acc':..., 'Dice' o 'mIoU': ...}
    """
    t_loss = [h['loss'] for h in hist_train]
    v_loss = [h['loss'] for h in hist_val]
    t_met  = [h[metric_key] for h in hist_train]
    v_met  = [h[metric_key] for h in hist_val]

    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].plot(t_loss, label='train'); ax[0].plot(v_loss, label='val')
    ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(True, alpha=.3)
    ax[1].plot(t_met,  label='train'); ax[1].plot(v_met,  label='val')
    ax[1].set_title(metric_key); ax[1].legend(); ax[1].grid(True, alpha=.3)
    plt.tight_layout(); plt.show()


def visualize_feature_maps(model, layer, xb, num_maps=16):
    """
    layer: módulo de PyTorch (por ejemplo, model.encoder[0] o model.down1.conv)
    xb: [1,C,H,W]
    """
    activ = {}
    def hook_fn(m, i, o): activ['maps'] = o.detach().cpu()
    handle = layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        _ = model(xb)

    handle.remove()
    maps = activ['maps'][0]  # [C,H,W]
    n = min(num_maps, maps.size(0))
    plt.figure(figsize=(n, 1.2))
    for i in range(n):
        plt.subplot(1,n,i+1)
        fmap = maps[i]
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-6)
        plt.imshow(fmap, cmap='gray'); plt.axis('off')
    plt.suptitle(f"{n} feature maps de {layer.__class__.__name__}")
    plt.show()


def occlusion_sensitivity(model, xb, yb, patch=16, stride=8, thr=0.5, device='cuda'):
    """
    Devuelve mapa con caída de Dice cuando se ocluye cada parche.
    xb: [1,C,H,W], yb: [1,1,H,W] o [1,H,W]
    """
    assert xb.size(0)==1
    model.eval()
    xb = xb.to(device); yb = yb.to(device)
    H, W = xb.shape[-2:]

    with torch.no_grad():
        base = torch.sigmoid(model(xb))[0,0]
        if yb.dim()==3: ybb = yb.unsqueeze(1).float()
        else:           ybb = yb.float()
        base_pred = (base>thr).float()
        inter = (base_pred*ybb[0]).sum()
        sums  = base_pred.sum() + ybb[0].sum()
        base_dice = (2*inter+1e-7)/(sums+1e-7)

    heat = np.zeros((H,W), dtype=float)
    for i in range(0, H-patch+1, stride):
        for j in range(0, W-patch+1, stride):
            xb_occ = xb.clone()
            xb_occ[..., i:i+patch, j:j+patch] = 0.0
            with torch.no_grad():
                p = torch.sigmoid(model(xb_occ))[0,0]
                pred = (p>thr).float()
                inter = (pred*ybb[0]).sum()
                sums  = pred.sum() + ybb[0].sum()
                d = (2*inter+1e-7)/(sums+1e-7)
            heat[i:i+patch, j:j+patch] = max(0.0, float(base_dice-d))

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.title("Imagen");
    img = xb[0].detach().cpu().permute(1,2,0).numpy();
    if img.max()>1: img=img/255.0
    plt.imshow(img); plt.axis('off')
    plt.subplot(1,3,2); plt.title("Pred prob"); plt.imshow(base.cpu(), vmin=0, vmax=1); plt.axis('off'); plt.colorbar(fraction=0.046)
    plt.subplot(1,3,3); plt.title("Occlusion ΔDice"); plt.imshow(heat); plt.axis('off'); plt.colorbar(fraction=0.046)
    plt.tight_layout(); plt.show()
    return heat


def collect_bottleneck(model, loader, device, hook_layer):
    """
    hook_layer: módulo en el bottleneck
    retorna: X [N, D] flattened, y opcional si loader lo provee
    """
    feats = []
    ys = []
    def hook_fn(m,i,o):
        feats.append(o.detach().cpu())
    h = hook_layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            _ = model(xb)
            if isinstance(yb, torch.Tensor):
                ys.append(yb.clone())
    h.remove()
    X = torch.cat([f.flatten(1) for f in feats], dim=0).numpy()
    Y = torch.cat(ys, dim=0).numpy() if ys else None
    return X, Y

def plot_embedding_2d(X, Y=None, method="umap", n=5000, title="Bottleneck embedding"):
    Xs = X[:n]
    if method=="umap" and HAVE_UMAP:
        reducer = umap.UMAP(n_components=2)
        Z = reducer.fit_transform(Xs)
    else:
        Z = TSNE(n_components=2, init='random', learning_rate='auto').fit_transform(Xs)
    plt.figure(figsize=(5,5))
    if Y is None:
        plt.scatter(Z[:,0], Z[:,1], s=2)
    else:
        y = Y[:n].reshape(-1)
        sc = plt.scatter(Z[:,0], Z[:,1], c=y, s=2, cmap='tab10')
        plt.colorbar(sc, fraction=0.046)
    plt.title(title); plt.axis('off'); plt.show()


def boundary_map(mask):
    mask = mask.astype(bool)
    return np.logical_xor(mask, ndi.binary_erosion(mask))

def boundary_f1(pred, gt, tol=2):
    """
    F1 sobre bordes con tolerancia en pixeles (afín a BFScore).
    pred, gt: binarios [H,W]
    """
    bp = boundary_map(pred); bg = boundary_map(gt)
    # dilatación para tolerancia
    se = ndi.generate_binary_structure(2,1)
    dp = ndi.binary_dilation(bp, structure=se, iterations=tol)
    dg = ndi.binary_dilation(bg, structure=se, iterations=tol)

    tp_p = (bp & dg).sum();  tp_g = (bg & dp).sum()
    p = bp.sum();            r = bg.sum()
    prec = tp_p / (p + 1e-7); rec = tp_g / (r + 1e-7)
    f1 = 2*prec*rec/(prec+rec+1e-7)
    return f1, prec, rec

def hausdorff_distance(pred, gt):
    """ Hausdorff simétrico aproximado (usa puntos de borde). """
    bp = np.argwhere(boundary_map(pred))
    bg = np.argwhere(boundary_map(gt))
    if len(bp)==0 or len(bg)==0:
        return np.nan
    d1 = directed_hausdorff(bp, bg)[0]
    d2 = directed_hausdorff(bg, bp)[0]
    return max(d1, d2)



