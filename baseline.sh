#Baseline models
python run.py --model gcn --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy
python run.py --model gin --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy
python run.py --model gat --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy
python run.py --model san --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy

python run.py --model gcn --dataset peptides-func --task graph --metric ap
python run.py --model gin --dataset peptides-func --task graph --metric ap
python run.py --model gat --dataset peptides-func --task graph --metric ap
python run.py --model san --dataset peptides-func --task graph --metric ap