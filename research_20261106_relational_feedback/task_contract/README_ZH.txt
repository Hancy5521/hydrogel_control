160组件元数据角色选择器（源已准备；不能把准备当执行）

select_roles.py仅允许读取模板列出的7份固定JSON元数据：原源合同、原eligible身份表、旧40/256/256/224/152五批角色收据。原材料gzip只保留路径/SHA指针，本程序不打开、不hash或解码它，不拟合模型。

排除928个完整来源组件后余7914；固定prefix relational_feedback_bma_onlyY_v1|加component_id做SHA256排序，取前160：32warm、32pool、96terminal，余7754。每组件选最小eligible原始行作代表，全部已知/未列入eligible的兄弟均保留该组件角色。新160全部是开发组，不因新划分变成最终独立确认。

执行前由root另存实际协议：复制SELECTION_PLAN_TEMPLATE.json，status填REGISTERED，registered_actual_utc填真实timezone-aware UTC，保持source/config/全部7个输入hash不变。批准JSON须approved=true，scope=METADATA_ONLY_RELATIONAL_FEEDBACK_ONLYY_SELECTION，并给出准确source_sha256与实际plan_sha256。

唯一执行入口：
python -X utf8 -B select_roles.py --plan <实际REGISTERED.json> --approval <实际ROOT_APPROVAL.json> --output-dir <尚不存在的selection_once>

输出ROLE_SELECTION.json及真实START/COMPLETE；出错保留FAILED_STOP，不覆盖或重选救援。此角色文件本身不授权新性质读取、查询、训练或终端评分；材料phase另需实际UTC注册和科学源码/hash批准。

共同学习信息成本352cell=32*3+32*2+96*2，每策略未来8次onlyY。96个终端Y仅供最终验证，另外记账。六策略重用同32池不扩充材料N，假想积分标签不是测量。此处没有执行上述未来科学步骤。
