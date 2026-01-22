rm -r data/mergeCols_check
mkdir data/mergeCols_check

head -n 1 data/incPrev281Conditions_Aurumv1draft20251219035947.csv | tr ',' '\n' | grep BD_MEDI | sed -n -e 's/^"\(.*\):.*$/\1/p' | sort > data/mergeCols_check/cols_aurum.txt
head -n 1 data/IncPrev281_GoldJan2024v6_fullDB20240612050700.csv | tr ',' '\n' | grep BD_MEDI | sed -n -e 's/^\(.*\):.*$/\1/p' | sort > data/mergeCols_check/cols_gold.txt

n_start=$(cat wdir.yml | grep -n mergeCols_AtoB | sed 's/\(^[0-9]*\):.*$/\1/')
n_start=$(($n_start + 1))

n_end=$(cat wdir.yml | grep -n combineLevels | sed 's/\(^[0-9]*\):.*$/\1/')
n_end=$(($n_end - 2))

cat wdir.yml | sed -n "${n_start},${n_end}p" | sed -n -e 's/^.*\(BD_MEDI.*\)",.*$/\1/p' | sort > data/mergeCols_check/mergeCols_aurum.txt
cat wdir.yml | sed -n "${n_start},${n_end}p" | sed -n -e 's/^.*\(BD_MEDI.*\)"]$/\1/p' | sort > data/mergeCols_check/mergeCols_gold.txt

diff data/mergeCols_check/cols_aurum.txt data/mergeCols_check/mergeCols_aurum.txt > data/mergeCols_check/mergeCols_diff_aurum.txt
diff data/mergeCols_check/cols_gold.txt data/mergeCols_check/mergeCols_gold.txt > data/mergeCols_check/mergeCols_diff_gold.txt
