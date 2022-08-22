# 一、数据集统计

| id | dataset | label | positive num | negative num | post num | avg comment num | remark |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| | Weibo | True | 2313 | 2351 | 4664 | 803.56 | |
| | Weibo(<600) | True | 2313 | 2351 | 4664 | 307.37 | 每个帖子最多使用600个评论 |
| | Weibo-2class | True | 3558 | 4072 | 7630 | 47.43 | |
| | Weibo-2class(>=20) | True | 1983 | 2532 | 4515 | 72.98 | 删除了低于20个评论的帖子 |
| | Weibo-unsup | False | | | 95260 | 45.01 | |