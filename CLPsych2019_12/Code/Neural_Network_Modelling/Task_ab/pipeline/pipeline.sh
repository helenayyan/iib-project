# Absolute path to this script, e.g. /home/user/bin/foo.sh
script=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
current_dir=$(dirname "$script")

read -p "Enter input text: " input_seq
echo ${input_seq} > "${current_dir}/input.txt"

preprocess_file="${current_dir}/preprocess.py"
input_file="${current_dir}/input.txt"
prefilter_file="${current_dir}/prefilter.py"
model_file="${current_dir}/best_model.py"
label_file="${current_dir}/get_label.py"
final_label_file="${current_dir}/label.txt"
dummy_file="${current_dir}/foo.txt"
csv_file="${current_dir}/result.csv"

rm $csv_file
touch $csv_file

start_time=`date +%s`

python $preprocess_file  --output_dir $current_dir  --raw_input $input_file > $dummy_file
python $prefilter_file  --output_dir $current_dir  --raw_input $input_file > $dummy_file

if [ -s $csv_file ];
then
    echo "---------------------prefiltered---------------------"
else
    
    python $model_file > $dummy_file
    echo "----------------------pass to model----------------------"
fi
echo "finish labeling. "

python $label_file
echo "User input: \n${input_seq}"

echo "The user is classified as: "
cat $final_label_file
echo " "
echo "run time is $(expr `date +%s` - $start_time) s"
