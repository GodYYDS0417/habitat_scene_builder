# sample_output —— demo 预设的真实产物

这是 aslam 上 `./run_scene_builder.sh demo` 跑出来的完整产物，
**`.glb` 已剔除**（stage 21.8MB + 6 个物体，纯属占地方），其余原样保留。

所以这份目录不能直接加载，它的作用是让你不跑代码就能看清：
- dataset config / stage config / object config / scene instance 长什么样
- semantic_id 是怎么分配的（`semantic_id_map.json`）
- 三个具身各自的 navmesh 大小差异（`navmeshes/`）
- 抽检图实际效果（`_preview/`，注意 semantic 图里每个物体一种颜色）
- `_preview/build_report.json` 里有本次构建的全部统计

补回 GLB 即可加载：把 stage GLB 放进 `stages/`、物体 GLB 放进 `objects/`，
文件名与同目录下的 `*_config.json` 里的 `render_asset` 字段对上就行。
