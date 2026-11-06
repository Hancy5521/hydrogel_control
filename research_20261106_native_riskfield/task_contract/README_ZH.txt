下一active-onlyY任务：metadata准备，未执行选择
===========================================

select_roles.py为新的独立source，不改刚封存的224组反馈实验。所有身份仍来自同一固定ELIGIBLE_METADATA和原full-sourcecomponent，不重建化学式分组、不读取或hash原生gzip。

父资料只作整组排除：legacy40＋此前两阶段各256＋已完成feedback224，共776个互斥component及其全部兄弟。后一个224的terminal已被消费，不再叫未见确认集，也不能回头挑其中好材料救分。

在剩8066组中固定排序：
  SHA256('native_riskfield_active_onlyY_v1|' + component_id)
按hex升序，再按component_id升序。取前152组：先64warm、再24pool、再64terminal；每组取最小eligible integer raw_row_zero_based代表。新152及全部兄弟同样是开发pilot资料，不冒充finalconfirmation；余7914组保持未选。

输出selection_once/ROLE_SELECTION.json，原identity字段不变：
  representatives.warm64 / pool24 / terminal64
  selected_groups：globalrank、role、representative、全部已知eligible_siblings
  all_prior_excluded_component_ids：776
  completed_feedback224_excluded_component_ids：刚完成的224
  new_development_consumed_component_ids：新152
  pool_sequence_component_ids / pool_sequence_jids：元数据hash原次序
  remaining_unselected_component_ids_in_registered_hash_order：7914。

未列出的ineligible兄弟无法在这个eligible-only metadata中枚举，但仍由完整source component排除/保留；不把已知eligible兄弟数说成全部原始结构数。

下一科学任务的费用计划只作为metadata说明，不是已经执行的披露：warm192、24pool双代理48、64terminal双代理128，共同setup368cell；之后每policy最多8个真实Y查询。多个策略在同一24pool上的counterfactual reuse只是一组材料的不同路径，不增加独立材料N或真实测量数。

只有root将SELECTION_PLAN_TEMPLATE登记为status REGISTERED和实际UTC，并精确绑定source与plan批准后，才运行唯一一次metadata选择：
  python -X utf8 -B select_roles.py --plan SELECTION_PLAN_REGISTERED.json
    --approval ROOT_SELECTION_APPROVAL.json --output-dir selection_once

审批字段：approved=true，scope=METADATA_ONLY_NATIVE_RISKFIELD_ONLYY_SELECTION，source_sha256和plan_sha256精确匹配。源只用标准库；prepare阶段只AST解析，没有import source、select新组、读新响应或运行模型。

真正获得pool/terminal代理、购买Y、预测、NIG更新与最终评分，需要另一个实际UTC/源码/输入/角色明确的科学协议，不能凭metadata role文件自行开始。有限原表完整案例选择的偏差和报告变量的物理限定仍保留，未重新证明MNAR、批次独立性或概率校准。
