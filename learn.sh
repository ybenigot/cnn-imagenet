timestamp=`date +%d%m%y-%H%M%S`
filename=record/test$timestamp.log
touch $filename
pidfile=`hostname`.pid
kill `cat $pidfile`

rm main.log
EPOCHS=200
echo start for $EPOCHS EPOCHS

python convnet4.py $EPOCHS $timestamp ${1:-NO} >> $filename 2>&1 & 

echo $! > $pidfile
tail -f $filename

