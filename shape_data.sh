python run.py --model gcn --datatype 3d --dataset psb --criterion weighted_cross_entropy --task graph
python run.py --model gin --datatype 3d --dataset psb --criterion weighted_cross_entropy --task graph
python run.py --model gat --datatype 3d --dataset psb --criterion weighted_cross_entropy --task graph
python run.py --model san --datatype 3d --dataset psb --criterion weighted_cross_entropy --task graph

python run.py --model gcn --datatype 3d --dataset psb --criterion weighted_cross_entropy --task graph --add_edges 1
python run.py --model gin --datatype 3d --dataset psb --criterion weighted_cross_entropy --task graph --add_edges 1
python run.py --model gat --datatype 3d --dataset psb --criterion weighted_cross_entropy --task graph --add_edges 1
python run.py --model san --datatype 3d --dataset psb --criterion weighted_cross_entropy --task graph --add_edges 1

python run.py --model gat --datatype 3d --dataset psb --criterion weighted_cross_entropy --task graph --partial 1 --k 6 --dropout 0.2 --space 10
