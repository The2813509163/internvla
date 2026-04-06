#!/bin/bash

# --- 最终解决方案 ---

# 1. 使用 `declare` 创建一个数组，这是最佳实践
declare -a DATASET_IDS_ARRAY

# 2. 使用 `mapfile` 将 `find` 的多行输出安全地读入数组的每个元素中
#    这种方法完全不受 IFS 变量的影响
mapfile -t DATASET_IDS_ARRAY < <(
  find -L "data/robotwin" -mindepth 2 -maxdepth 2 -type d -name "aloha-agilex*" 2>/dev/null |
  while read -r d; do
    if [[ -d "$d/meta" && -d "$d/videos" ]]; then
      echo "${d#data/}"
    fi
  done |
  sort -u
)

# 3. 检查数组是否成功填充
if [ ${#DATASET_IDS_ARRAY[@]} -eq 0 ]; then
    echo "错误：未能找到任何数据集。请检查 'data/robotwin' 目录的路径和内容。"
    exit 1
fi

echo "成功找到 ${#DATASET_IDS_ARRAY[@]} 个数据集，准备计算统计数据..."
# echo "数据集列表: ${DATASET_IDS_ARRAY[@]}" # 如果你想看列表，可以取消这行的注释

# 4. 执行 Python 脚本
python util_scripts/compute_norm_stats_multi.py \
  --action_mode delta \
  --chunk_size 50 \
  --repo_ids "${DATASET_IDS_ARRAY[@]}"  # <-- (修复1) 使用正确的复数 --repo_ids
                                        # <-- (修复2) 使用 "${ARRAY[@]}" 语法安全地展开数组