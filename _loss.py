import jittor as jt
import jittor.nn as nn


class EnhancedBCEFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0,
                 adaptive_gamma=True, class_weights=None,
                 smooth_focus=False, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.adaptive_gamma = adaptive_gamma
        self.class_weights = class_weights
        self.smooth_focus = smooth_focus
        self.size_average = size_average

        if adaptive_gamma:
            self.base_gamma = gamma
            self.gamma = jt.array([gamma], dtype=jt.float32)

    def execute(self, preds, targets):
        bce_loss = nn.binary_cross_entropy_with_logits(
            preds, targets,
            weight=None,
            size_average=False
        )
        probas = jt.sigmoid(preds)

        # 动态调整gamma
        if self.adaptive_gamma:
            confidence = jt.abs(probas - 0.5) * 2
            gamma = self.base_gamma * (1.0 + confidence)
        else:
            gamma = self.gamma

        # Focal Loss调制因子
        p_t = jt.ternary(targets == 1, probas, 1 - probas)

        modulating_factor = (1 - p_t) ** gamma

        # 平滑聚焦
        if self.smooth_focus:
            correct_mask = (probas > 0.5) == (targets > 0.5)
            focus_factor = jt.ternary(
                correct_mask,
                jt.full_like(probas, 0.5),
                jt.full_like(probas, 1.0)
            )
        else:
            focus_factor = jt.ones_like(probas)

        # 最终损失
        loss = bce_loss * modulating_factor * focus_factor

        # 类别权重
        if self.class_weights is not None:
            weight_mask = jt.ternary(
                targets == 1,
                jt.full_like(targets, self.class_weights[1]),
                jt.full_like(targets, self.class_weights[0])
            )
            loss = loss * weight_mask

        # alpha平衡
        if self.alpha is not None:
            alpha_factor = jt.ternary(
                targets == 1,
                jt.full_like(targets, self.alpha),
                jt.full_like(targets, 1 - self.alpha)
            )
            loss = loss * alpha_factor
        elif self.alpha is None:  # 自动平衡
            num_pos = (targets == 1).sum()
            num_neg = (targets == 0).sum()
            alpha = num_neg / (num_pos + num_neg + 1e-7)
            alpha_factor = jt.ternary(
                targets == 1,
                jt.full_like(targets, alpha),
                jt.full_like(targets, 1 - alpha)
            )
            loss = loss * alpha_factor

        if self.size_average:
            return loss.mean()
        return loss.sum()


class CombinedBCEFocalLoss(nn.Module):
    def __init__(self, focal_kwargs=None, bce_weight=0.5, focal_weight=0.5,
                 adaptive_weight=False, size_average=True):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.adaptive_weight = adaptive_weight
        self.size_average = size_average

        self.focal_loss = EnhancedBCEFocalLoss(
            **(focal_kwargs if focal_kwargs else {}),
            size_average=False
        )
        self.bce_loss = nn.BCEWithLogitsLoss(size_average=False)

        if adaptive_weight:
            self.bce_weight = jt.array([bce_weight], dtype=jt.float32)
            self.focal_weight = jt.array([focal_weight], dtype=jt.float32)

    def _calculate_adaptive_weights(self, pred, target):
        if self.adaptive_weight:
            probas = jt.sigmoid(pred)
            difficulty = 2 * jt.abs(probas - 0.5).mean()  # 范围[0,1]

            # 动态调整（保持总权重为1）
            focal_weight = 0.5 + 0.5 * difficulty  # 难样本增加Focal权重
            bce_weight = 1 - focal_weight

            return bce_weight, focal_weight
        return self.bce_weight, self.focal_weight

    def execute(self, pred, target):
        loss_bce = self.bce_loss(pred, target)
        loss_focal = self.focal_loss(pred, target)
        #
        # print(f"Raw BCE Loss: {loss_bce.mean().item():.4f}")
        # print(f"Raw Focal Loss: {loss_focal.mean().item():.4f}")

        bce_w, focal_w = self._calculate_adaptive_weights(pred, target)
        # print(f"Adaptive Weights - BCE: {bce_w.item():.4f}, Focal: {focal_w.item():.4f}")

        combined_loss = bce_w * loss_bce + focal_w * loss_focal

        if self.size_average:
            return combined_loss.mean()
        return combined_loss.sum()

if __name__  == "__main__":
    loss_fn = CombinedBCEFocalLoss(
        focal_kwargs={
            'alpha': 0.25,
            'gamma': 2.0,
            'adaptive_gamma': False,
            'smooth_focus': False
        },
        adaptive_weight=True,
        size_average=True
    )

    pred = jt.randn(4, 1)  # logits
    label = jt.float32([[1], [0], [1], [1]])  # gt

    loss = loss_fn(pred, label)
    print(f"Combined Loss: {loss}")