PyMICPATH=/home/disk4t/projects/PyMIC_project/PyMIC
export PYTHONPATH=$PYTHONPATH:$PyMICPATH

# python run.py train config/psseg_word.cfg
# python run.py test config/psseg_word.cfg
# python $PyMICPATH/pymic/util/evaluation_seg.py --metric dice --cls_num 8 \
#   --gt_dir ./data/Word_cropWL/labelsTs \
#   --seg_dir result/word_psseg

# python $PyMICPATH/pymic/net_run/train.py config/psseg_word_2d.cfg
# python $PyMICPATH/pymic/net_run/predict.py config/psseg_word_2d.cfg
python $PyMICPATH/pymic/util/evaluation_seg.py --metric dice --cls_num 8 \
  --gt_dir ./data/Word_cropWL/labelsTs \
  --seg_dir result/word_psseg_2d