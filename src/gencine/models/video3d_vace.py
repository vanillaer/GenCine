"""Geometry-conditioned VACE blocks."""
import torch
from .wan_dit import DiTBlock, precompute_freqs_cis_3d


class VaceWanAttentionBlock(DiTBlock):
    def __init__(
        self,
        has_image_input,
        dim,
        num_heads,
        ffn_dim,
        eps=1e-6,
        block_id=0,
        temporal_factor=2,
        spatial_factor=2,
    ):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        self.temporal_factor = temporal_factor
        self.spatial_factor = spatial_factor
        kernel_size, stride, padding = self.build_proj_params(
            self.temporal_factor, self.spatial_factor
        )
        if block_id == 0:
            self.before_proj = self.build_before_proj(
                kernel_size, stride, padding
            )
            torch.nn.init.dirac_(self.before_proj.weight)
            torch.nn.init.zeros_(self.before_proj.bias)
        self.after_proj = self.build_after_proj(
            kernel_size, stride, padding
        )
        torch.nn.init.zeros_(self.after_proj.weight)
        torch.nn.init.zeros_(self.after_proj.bias)

    def build_before_proj(self, kernel_size, stride, padding):
        return torch.nn.Conv3d(
            self.dim,
            self.dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def build_after_proj(self, kernel_size, stride, padding):
        # output_padding is resolved per forward from the requested main grid (see
        # forward via output_size), so one set of weights targets any resolution.
        return torch.nn.ConvTranspose3d(
            self.dim,
            self.dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    @staticmethod
    def build_proj_params(temporal_factor, spatial_factor):
        kernel_size = (
            1 if temporal_factor == 1 else 3,
            spatial_factor,
            spatial_factor,
        )
        stride = (temporal_factor, spatial_factor, spatial_factor)
        padding = (
            0 if temporal_factor == 1 else 1,
            0,
            0,
        )
        return kernel_size, stride, padding

    @staticmethod
    def grid_to_tokens(x):
        return x.flatten(2).transpose(1, 2)

    @staticmethod
    def tokens_to_grid(x, grid_size):
        f, h, w = grid_size
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], f, h, w)

    @staticmethod
    def conv_output_size(input_size, kernel_size, stride, padding):
        return tuple(
            (size + 2 * pad - kernel) // step + 1
            for size, kernel, step, pad in zip(input_size, kernel_size, stride, padding)
        )

    @staticmethod
    def conv_transpose_output_size(input_size, kernel_size, stride, padding, output_padding=(0, 0, 0)):
        return tuple(
            (size - 1) * step - 2 * pad + kernel + out_pad
            for size, kernel, step, pad, out_pad in zip(input_size, kernel_size, stride, padding, output_padding)
        )

    def validate_grid_sizes(self, main_grid_size, side_grid_size):
        kernel_size, stride, padding = self.build_proj_params(self.temporal_factor, self.spatial_factor)
        if self.block_id == 0:
            expected_side = self.conv_output_size(main_grid_size, kernel_size, stride, padding)
            assert expected_side == side_grid_size, (
                f"before_proj expects side grid {expected_side}, but got {side_grid_size} "
                f"for main grid {main_grid_size}."
            )
        # Conv downsampling floors, so each side dim maps back to main 2N or 2N+1;
        # forward selects the exact target via output_size. Here we only check it is
        # reachable (needed output_padding within [0, stride) per axis).
        base_main = self.conv_transpose_output_size(side_grid_size, kernel_size, stride, padding)
        output_padding = tuple(m - b for m, b in zip(main_grid_size, base_main))
        assert all(0 <= op < s for op, s in zip(output_padding, stride)), (
            f"after_proj cannot reach main grid {main_grid_size} from side grid {side_grid_size}: "
            f"needs output_padding {output_padding} outside [0, stride={stride})."
        )

    def forward(self, c, x, context, t_mod, freqs, side_grid_size, main_grid_size):
        self.validate_grid_sizes(main_grid_size, side_grid_size)
        if self.block_id == 0:
            x_side = self.before_proj(self.tokens_to_grid(x, main_grid_size))
            c = self.grid_to_tokens(c + x_side)
        c = super().forward(c, context, t_mod, freqs)
        c_skip = self.after_proj(self.tokens_to_grid(c, side_grid_size), output_size=main_grid_size)
        c_skip = self.grid_to_tokens(c_skip)
        return c, c_skip


class VaceWanModel(torch.nn.Module):
    def __init__(
        self,
        vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        vace_in_dim=60,
        patch_size=(1, 2, 2),
        temporal_factor=1,  
        spatial_factor=2,
        has_image_input=False,
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        eps=1e-6,
    ):
        super().__init__()
        self.vace_layers = vace_layers
        self.vace_in_dim = vace_in_dim
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}
        with torch.device("cpu"):
            self.freqs = precompute_freqs_cis_3d(dim // num_heads)

        # vace blocks
        self.vace_blocks = torch.nn.ModuleList([
            VaceWanAttentionBlock(
                has_image_input,
                dim,
                num_heads,
                ffn_dim,
                eps,
                block_id=i,
                temporal_factor=temporal_factor,
                spatial_factor=spatial_factor,
            )
            for i in self.vace_layers
        ])

        # Video3D VACE input layout:
        # [pts_video(16), pts_mask(4), disp_video(16), disp_mask(4), partial_id(16), partial_disp_mask(4)]
        self.vace_patch_embedding = torch.nn.Conv3d(vace_in_dim, dim, kernel_size=patch_size, stride=patch_size)

    @staticmethod
    def grid_token_count(grid_size):
        f, h, w = grid_size
        return f * h * w

    @classmethod
    def split_main_tokens(cls, x, main_grid_size_full, main_grid_size_video, main_prefix_frames):
        assert main_grid_size_full[1:] == main_grid_size_video[1:], (
            f"main full grid {main_grid_size_full} and video grid {main_grid_size_video} "
            "must share spatial dimensions."
        )
        assert main_grid_size_full[0] == main_grid_size_video[0] + main_prefix_frames, (
            f"main full grid {main_grid_size_full} must equal video grid {main_grid_size_video} "
            f"plus {main_prefix_frames} prefix frames."
        )
        full_token_count = cls.grid_token_count(main_grid_size_full)
        video_token_count = cls.grid_token_count(main_grid_size_video)
        prefix_token_count = main_prefix_frames * main_grid_size_full[1] * main_grid_size_full[2]
        assert x.shape[1] == full_token_count, (
            f"x token count {x.shape[1]} must match main full grid token count {full_token_count}."
        )
        assert full_token_count == prefix_token_count + video_token_count, (
            f"main grid token split mismatch: full={full_token_count}, "
            f"prefix={prefix_token_count}, video={video_token_count}."
        )
        x_video = x[:, prefix_token_count:]
        return x_video, prefix_token_count

    @staticmethod
    def pad_prefix_tokens(x, prefix_token_count):
        if prefix_token_count == 0:
            return x
        prefix = x.new_zeros(x.shape[0], prefix_token_count, x.shape[2])
        return torch.cat([prefix, x], dim=1)

    def forward(
        self, x, vace_context, context, t_mod, freqs,
        main_grid_size_full,
        main_grid_size_video=None,
        main_prefix_frames: int = 0,
    ):
        if main_grid_size_video is None:
            main_grid_size_video = main_grid_size_full
        x_video, prefix_token_count = self.split_main_tokens(
            x, main_grid_size_full, main_grid_size_video, main_prefix_frames
        )
        c = self.vace_patch_embedding(vace_context)
        side_grid_size = tuple(c.shape[2:])
        cf, ch, cw = side_grid_size
        side_freqs = torch.cat([
            self.freqs[0][:cf].view(cf, 1, 1, -1).expand(cf, ch, cw, -1),
            self.freqs[1][:ch].view(1, ch, 1, -1).expand(cf, ch, cw, -1),
            self.freqs[2][:cw].view(1, 1, cw, -1).expand(cf, ch, cw, -1),
        ], dim=-1).reshape(cf * ch * cw, 1, -1).to(x.device)
        hints = []

        for block in self.vace_blocks:
            c, c_skip = block(
                c, x_video, context, t_mod, side_freqs, side_grid_size, main_grid_size_video,
            )
            hints.append(self.pad_prefix_tokens(c_skip, prefix_token_count))
        return hints
