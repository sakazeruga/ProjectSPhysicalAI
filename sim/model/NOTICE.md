# Franka Emika Panda モデル

`panda_care.xml` と `assets/` は [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
の `franka_emika_panda/panda_nohand.xml` (commit b0655057, Apache License 2.0, 全文は `LICENSE`) に由来する。

本プロジェクトでの変更点:

- 位置アクチュエータをトルクモータへ置換 (力制御のため)
- 先端にケアパッド (`care_pad`) と接触力センサ (`pad_force`) を追加
- 未使用のメッシュを削除
