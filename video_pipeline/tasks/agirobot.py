from __future__ import annotations
from typing import Any, Dict, List

from .base import Task
from .registry import register_task

# -------------------------- 移植的 Prompt 模板 --------------------------

ACTION_PROMPT_TEMPLATE = """
请详细分析视频片段，基于原始标注：{raw_text}，禁止使用括号，严格按照以下要求进行机器人动作的结构化描述：
1. 时序逻辑清晰化：采用 First... Then... Next... Meanwhile... Finally... 等过渡连词，将动作拆解为符合物理逻辑的分步流程，明确动作发生的先后顺序与因果关联。
2. 动作细节具象化：避免抽象词汇，将动作转化为可视觉化的具体细节，句式严格遵循“主体+核心动作+辅助动作/姿态+接触部位”的结构，例如“机器人机械臂先缓慢抬起，随后前臂轻微弯曲，末端夹具精准夹持前景中的零件”。
3. 空间方位精准化：使用 Left/Right side of the frame、画面左侧/右侧、Top/Bottom、顶部/底部、Center、中心、Foreground/Background、前景/背景、Relative to [object]等方位术语，明确动作发生的空间位置及主体与周边物体的相对关系。
4. 多主体区分明确化：若画面存在多个机器人或物体，通过颜色，如灰色工业机器人、蓝色协作机器人、形态特征，如带旋转云台的机器人、多关节机械臂机器人、空间位置等组合属性进行唯一区分，避免主体混淆。
5. 量化描述相对化：不使用具体的距离、角度、速度数值，改用相对描述，如 轻微、缓慢、适度、大幅、平稳、匀速 等，确保描述的客观性与通用性。
6. 动作连贯性保障：确保各动作步骤之间衔接自然，符合机器人运动的机械原理，避免出现关节运动异常、动作断层等逻辑矛盾。
7. 禁止使用括号：全程禁止使用任何形式的括号，包括小括号（）、圆括号等所有括号类符号，所有补充说明内容均直接融入语句，不得通过括号标注。如：First，“画面左侧的左臂机械臂（末端带黑色夹具）缓慢移动”应该为“画面左侧的末端带黑色夹具的左臂机械臂”。
要求：描述语言专业流畅，聚焦机器人动作本身，以视频生成prompt的语境进行叙述，而非仅仅对场景进行描述，字数不少于100字，仅输出动作描述文本，无需额外解释。
"""

SCENE_PROMPT_TEMPLATE = """
请结合视频首帧图像、原始初始场景描述：{raw_text}，以及第一个动作的标注信息：{first_action_text}，禁止使用括号，严格按照以下要求生成适配第一个动作对应视频的完整生成prompt：
1. 充分利用原始场景描述：若原始场景描述中有明确的物品名称，则不要在描述中出现模糊性叙述，而应替换成相应的准确物品名称，如“鱼油”而非“瓶装液体”。
2. 空间布局体系化：以“整体场景定位→前景元素分布→中景元素分布→背景元素分布”的逻辑展开，使用 Left/Right side of the frame（画面左侧/右侧）、Top/Bottom（顶部/底部）、Center（中心）、Foreground/Background（前景/背景）、Adjacent to（相邻）、Symmetrically distributed（对称分布）等方位术语，清晰呈现所有物体的空间排布关系。
3. 多对象区分精准化：若存在多个机器人或物体，通过“颜色属性（如银色金属框架、黄色警示标识）、形态特征（如圆柱形工件、方形操作台）、功能标识（如带有‘抓取区’字样的平台）、空间位置”的组合维度进行区分，明确标注各对象之间的相对位置（如“前景中心的协作机器人右侧，相邻放置着若干中等大小的圆柱形工件”）。
4. 视觉细节客观化：避免抽象词汇，将场景元素转化为具体视觉细节，句式遵循“主体+空间位置+形态特征+表面状态/纹理”的结构，例如“画面背景右侧的金属操作台，表面带有细微划痕，边缘设有黑色防护栏”。
5. 镜头视角与朝向明确化：明确标注镜头的位置与视角类型，包括俯视平视仰视侧视等视角表述，清晰说明镜头与核心物体的位置关系，例如镜头正对前景中心的协作机器人镜头从画面右侧斜向拍摄中景的工件摆放区等，同时精准描述各物体的朝向特征，例如机器人机械臂朝向画面下方的工件台圆柱形工件轴向平行于画面横向金属操作台台面朝向镜头等。
6. 量化描述模糊化：不描述具体的尺寸、重量、精确数量数值，改用相对描述（如 若干、少数、中等大小、大型、小型、平整、粗糙 等）。
7. 动作场景融合化：将第一个动作的核心逻辑、时序流程、主体动作特征自然融入场景描述中，明确动作发生的起始状态、主体位置、动作轨迹与场景元素的交互关系，确保生成的prompt能精准匹配第一个动作对应视频的视觉呈现需求。
8. 场景用途推理合理化：基于场景布局、物体特征（如工件类型、机器人功能、操作平台结构）和第一个动作的特征，推测场景的具体工业用途（如 零件装配区、工件分拣区、设备调试区 等），推理过程需隐含在场景描述中，不单独罗列。
9. 环境氛围补充：简要描述场景的环境特征（如 室内工业厂房、光线均匀、无明显杂物 等），增强场景描述的完整性，同时匹配动作发生的环境氛围。
10. 格式严格合规：全程禁止使用任何形式的括号，包括小括号、圆括号等所有括号类符号，所有补充说明内容均直接融入语句，不得通过括号标注。
要求：描述语言专业流畅，逻辑层次清晰，既包含完整的场景信息，又精准适配第一个动作的视频生成需求，字数不少于200字，仅输出视频生成prompt形式的描述文本，无需额外解释。
"""

@register_task
class AgiRobotActionTask(Task):
    name = "agirobot_action"

    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        # sample 会包含: {"path": "segment.mp4", "raw_text": "原始动作描述"}
        raw_text = sample.get("raw_text", "无动作描述")
        prompt = ACTION_PROMPT_TEMPLATE.format(raw_text=raw_text)
        
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video", "video": sample["__video_uri"]},
                ],
            },
        ]

    def parse(self, generated_text: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {"detailed_action_text": generated_text.strip()}

@register_task
class AgiRobotSceneTask(Task):
    name = "agirobot_scene"

    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        # sample 会包含: {"path": "speedup_video_frame.jpg" (或者 mp4), "raw_text": "场景描述", "first_action_text": "..."}
        # 注意：这里我们传入图片或视频首帧。框架会自动处理 sample["__video_uri"]
        
        raw_text = sample.get("raw_text", "无场景描述")
        first_action_text = sample.get("first_action_text", "无首个动作描述")
        
        prompt = SCENE_PROMPT_TEMPLATE.format(
            raw_text=raw_text,
            first_action_text=first_action_text
        )
        
        msg_content = [{"type": "text", "text": prompt}]
        
        # 判断是视频还是图片（Dataset层处理）
        if sample.get("is_image", False):
             msg_content.append({"type": "image", "image": sample["__video_uri"]})
        else:
             msg_content.append({"type": "video", "video": sample["__video_uri"]})

        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": msg_content},
        ]

    def parse(self, generated_text: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {"detailed_init_scene_text": generated_text.strip()}
