#!/usr/bin/env python3
"""
MCAP 话题重命名工具

功能:
- 读取 MCAP 文件
- 重命名指定的话题
- 保存到新的 MCAP 文件
- 完整保留所有数据、元数据和压缩设置

使用方法:
    python mcap_topic_renamer.py input.mcap output.mcap --rename /old_topic:/new_topic

    # 重命名多个话题
    python mcap_topic_renamer.py input.mcap output.mcap \
        --rename /old_topic1:/new_topic1 \
        --rename /old_topic2:/new_topic2

依赖安装:
    pip install mcap
"""

import argparse
import sys
from pathlib import Path
from typing import Dict
from mcap.reader import make_reader
from mcap.writer import Writer


def rename_topics_in_mcap(
    input_path: Path,
    output_path: Path,
    topic_mapping: Dict[str, str],
    verbose: bool = False
):
    """
    重命名 MCAP 文件中的话题

    Args:
        input_path: 输入 MCAP 文件路径
        output_path: 输出 MCAP 文件路径
        topic_mapping: 话题映射字典 {原话题名: 新话题名}
        verbose: 是否显示详细信息
    """

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 统计信息
    stats = {
        'total_messages': 0,
        'renamed_messages': 0,
        'topics_found': set(),
        'topics_renamed': set(),
        'schemas': 0,
        'channels': 0,
        'attachments': 0,
        'metadata': 0,
    }

    # Channel ID 映射（旧ID -> 新ID）
    channel_id_mapping = {}
    # Schema ID 映射
    schema_id_mapping = {}

    print(f"📖 读取输入文件: {input_path}")
    print(f"   大小: {input_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"✏️  重命名规则:")
    for old_topic, new_topic in topic_mapping.items():
        print(f"   {old_topic} → {new_topic}")
    print()

    with open(input_path, "rb") as input_file:
        reader = make_reader(input_file)

        with open(output_path, "wb") as output_file:
            writer = Writer(output_file)
            writer.start()

            # 获取 summary 以访问 schemas 和 channels
            summary = reader.get_summary()

            if summary is None:
                raise ValueError("无法读取 MCAP 文件摘要信息")

            print(f"📋 处理 schemas 和 channels...")

            # 第一步: 复制 Schema 信息
            if summary.schemas:
                for schema_id, schema in summary.schemas.items():
                    new_schema_id = writer.register_schema(
                        name=schema.name,
                        encoding=schema.encoding,
                        data=schema.data
                    )
                    schema_id_mapping[schema_id] = new_schema_id
                    stats['schemas'] += 1

                    if verbose:
                        print(f"📋 Schema: {schema.name} (ID: {schema_id} → {new_schema_id})")

            # 第二步: 复制并重命名 Channel 信息
            if summary.channels:
                for channel_id, channel in summary.channels.items():
                    old_topic = channel.topic
                    new_topic = topic_mapping.get(old_topic, old_topic)

                    stats['topics_found'].add(old_topic)

                    # 获取新的 schema ID
                    new_schema_id = schema_id_mapping.get(channel.schema_id, channel.schema_id)

                    # 注册新的 channel
                    new_channel_id = writer.register_channel(
                        topic=new_topic,
                        message_encoding=channel.message_encoding,
                        schema_id=new_schema_id,
                        metadata=channel.metadata
                    )

                    channel_id_mapping[channel_id] = new_channel_id
                    stats['channels'] += 1

                    if old_topic != new_topic:
                        stats['topics_renamed'].add(old_topic)
                        if verbose:
                            print(f"✏️  Channel: {old_topic} → {new_topic} (ID: {channel_id} → {new_channel_id})")
                    else:
                        if verbose:
                            print(f"📡 Channel: {old_topic} (ID: {channel_id} → {new_channel_id})")

            print()

            # 第三步: 复制消息数据
            print("📦 正在复制消息数据...")
            for schema, channel, message in reader.iter_messages():
                stats['total_messages'] += 1

                # 获取新的 channel ID
                new_channel_id = channel_id_mapping.get(channel.id)

                if new_channel_id is None:
                    print(f"⚠️  警告: 未找到 channel ID {channel.id} 的映射，跳过消息")
                    continue

                # 检查是否是重命名的话题
                if channel.topic in topic_mapping:
                    stats['renamed_messages'] += 1

                # 写入消息（使用新的 channel ID）
                writer.add_message(
                    channel_id=new_channel_id,
                    log_time=message.log_time,
                    data=message.data,
                    publish_time=message.publish_time,
                    sequence=message.sequence
                )

                # 进度显示
                if stats['total_messages'] % 1000 == 0:
                    print(f"  已处理 {stats['total_messages']} 条消息...", end='\r')

            # 第四步: 复制附件
            print(f"\n📎 正在复制附件...")
            for attachment in reader.iter_attachments():
                writer.add_attachment(
                    log_time=attachment.log_time,
                    create_time=attachment.create_time,
                    name=attachment.name,
                    media_type=attachment.media_type,
                    data=attachment.data
                )
                stats['attachments'] += 1

                if verbose:
                    print(f"📎 Attachment: {attachment.name}")

            # 第五步: 复制元数据
            print(f"ℹ️  正在复制元数据...")
            for metadata in reader.iter_metadata():
                writer.add_metadata(
                    name=metadata.name,
                    data=metadata.metadata
                )
                stats['metadata'] += 1

                if verbose:
                    print(f"ℹ️  Metadata: {metadata.name}")

            writer.finish()

    # 显示统计信息
    print("\n")
    print("=" * 70)
    print("✅ 处理完成!")
    print("=" * 70)
    print(f"📊 统计信息:")
    print(f"   Schemas: {stats['schemas']}")
    print(f"   Channels: {stats['channels']}")
    print(f"   总消息数: {stats['total_messages']}")
    print(f"   重命名消息数: {stats['renamed_messages']}")
    print(f"   附件: {stats['attachments']}")
    print(f"   元数据: {stats['metadata']}")
    print(f"   发现话题数: {len(stats['topics_found'])}")
    print(f"   重命名话题数: {len(stats['topics_renamed'])}")
    print()
    print(f"📁 输出文件: {output_path}")
    print(f"   输入大小: {input_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"   输出大小: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"   大小比率: {(output_path.stat().st_size / input_path.stat().st_size * 100):.1f}%")
    print("=" * 70)

    if stats['topics_renamed']:
        print()
        print("✏️  已重命名的话题:")
        for topic in sorted(stats['topics_renamed']):
            print(f"   {topic} → {topic_mapping[topic]}")

    if stats['topics_found']:
        print()
        print("📡 所有话题:")
        for topic in sorted(stats['topics_found']):
            if topic in topic_mapping:
                print(f"   {topic} → {topic_mapping[topic]} (重命名)")
            else:
                print(f"   {topic} (保持不变)")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="MCAP 话题重命名工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 重命名单个话题
  %(prog)s input.mcap output.mcap --rename /old_topic:/new_topic

  # 重命名多个话题
  %(prog)s input.mcap output.mcap \\
      --rename /camera/image:/camera/rgb/image \\
      --rename /joint_states:/robot/joint_states

  # 重命名 rosbag 目录中的 MCAP 文件
  %(prog)s rosbag_data/teaching_20240130/teaching_20240130_0.mcap \\
      output.mcap \\
      --rename /ee_camera/rgb/image_raw:/observation.images.ee_cam \\
      --rename /external_camera/rgb/image_raw:/observation.images.ext_cam
        """
    )

    parser.add_argument(
        'input',
        type=str,
        help='输入 MCAP 文件路径'
    )

    parser.add_argument(
        'output',
        type=str,
        help='输出 MCAP 文件路径'
    )

    parser.add_argument(
        '-r', '--rename',
        action='append',
        dest='renames',
        metavar='OLD:NEW',
        help='话题重命名规则 (格式: /old_topic:/new_topic)，可多次使用'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细信息'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='只显示将要执行的操作，不实际修改文件'
    )

    args = parser.parse_args()

    # 检查参数
    if not args.renames:
        parser.error("至少需要指定一个重命名规则 (--rename)")

    # 解析重命名规则
    topic_mapping = {}
    for rename_rule in args.renames:
        if ':' not in rename_rule:
            parser.error(f"无效的重命名规则格式: '{rename_rule}'，应为 'OLD:NEW'")

        old_topic, new_topic = rename_rule.split(':', 1)
        old_topic = old_topic.strip()
        new_topic = new_topic.strip()

        if not old_topic or not new_topic:
            parser.error(f"话题名不能为空: '{rename_rule}'")

        topic_mapping[old_topic] = new_topic

    # 转换路径
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    # Dry run 模式
    if args.dry_run:
        print("🔍 DRY RUN 模式 - 不会修改文件")
        print(f"📖 输入文件: {input_path}")
        print(f"📝 输出文件: {output_path}")
        print(f"✏️  重命名规则:")
        for old_topic, new_topic in topic_mapping.items():
            print(f"   {old_topic} → {new_topic}")
        return

    # 执行重命名
    try:
        rename_topics_in_mcap(
            input_path=input_path,
            output_path=output_path,
            topic_mapping=topic_mapping,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"\n❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
