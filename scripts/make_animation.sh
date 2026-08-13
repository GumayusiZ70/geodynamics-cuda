#!/usr/bin/env bash

# 将程序输出的六位补零 PNG 快照按时间顺序合成为 GIF 或 MP4。
# 用法：scripts/make_animation.sh [输入目录] [输出文件]
set -euo pipefail

input_directory="${1:-output}"
output_file="${2:-${input_directory}/temperature.gif}"
frames_per_second="${FPS:-5}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "未找到 ffmpeg。请先安装 ffmpeg，再重新执行本脚本。" >&2
    exit 1
fi

if [[ ! -d "${input_directory}" ]]; then
    echo "输入目录不存在: ${input_directory}" >&2
    exit 1
fi

if [[ "${output_file}" == *.mp4 ]]; then
    ffmpeg -y -framerate "${frames_per_second}" -pattern_type glob \
        -i "${input_directory}/temperature_*.png" \
        -vf "scale=1024:-1:flags=nearest" -pix_fmt yuv420p "${output_file}"
else
    ffmpeg -y -framerate "${frames_per_second}" -pattern_type glob \
        -i "${input_directory}/temperature_*.png" \
        -vf "fps=${frames_per_second},scale=1024:-1:flags=nearest" "${output_file}"
fi

echo "动画已生成: ${output_file}"
