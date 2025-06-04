import json
import re

input_file = 'unified_feedback_stage_1_600k.json'
output_file = 'unified_feedback_stage_1_600k_clean.json'

cleaned_data = []
error_count = 0

def extract_json_objects(text):
    """使用正则提取 JSON 对象（仅限对象而非数组）"""
    pattern = re.compile(r'\{.*?\}', re.DOTALL)
    return pattern.findall(text)

# 用二进制方式读取，忽略非法字符
with open(input_file, 'rb') as f:
    raw_bytes = f.read()

# 尝试用 'utf-8' 解码，忽略错误字节（可能丢失少量信息）
text = raw_bytes.decode('utf-8', errors='ignore')

# 提取出所有 JSON 对象（假设顶层是对象组成的数组）
json_objects = extract_json_objects(text)

for i, obj_str in enumerate(json_objects):
    try:
        item = json.loads(obj_str)
    except Exception as e:
        print(f"跳过第 {i+1} 个对象，解析失败：{e}")
        error_count += 1
        continue
    if (len(item.keys()) == 3) and ("output" in item.keys()) and ("input" in item.keys()) and ("instruction" in item.keys()):
        # 处理对应数据
        try:
            item["instruction"] = r"[User Question]\n" + item["instruction"] + r"\n\n"
            item["output"] = item["output"].replace("Response A: ", r"[The Start of Assistant A's Answer]\n")
            item["output"] = item["output"].replace("\n\nResponse B: ", r"[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n")
            item["output"] = item["output"] + r"\n[The End of Assistant B's Answer]"
        except:
            import pdb; pdb.set_trace()

        cleaned_data.append(item)

# 输出为标准 JSON 数组
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"\n✅ 清洗完成：共保留 {len(cleaned_data)} 个对象，跳过 {error_count} 个非法对象。")

