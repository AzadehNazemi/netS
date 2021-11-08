#convert png to raw or bin file in bash script
for al in $(ls  |grep png)
do 

    dest=$(echo $al|sed 's/png/bin/g')
    echo $dest

    bytes=$(identify -format "%[fx:h*w*2]" $al)
convert $al -depth 16 pgm:- | tail -c $bytes >  $dest
done
