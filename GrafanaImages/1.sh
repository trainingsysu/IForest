#!/bin/bash

url="http://127.0.0.1:3000/d/nJ8T4UMZk/zhou-si-yi-chang?orgId=1&fullscreen&panelId=2&tab=metrics&refresh=5s&from="
currentStamp=`date +%s`
to="&to="
fromStamp=`date -d "5 minutes ago" +%s`
currentTimeStamp=$((currentStamp*1000+`date "+%N"`/1000000))
fromTimeStamp=$((fromStamp*1000+`date "+%N"`/1000000))
size="&width=1000&height=500&tz=Asia%2FShanghai"

str1="$url$fromTimeStamp$to$currentTimeStamp$size"

curl -H "Authorization: Bearer eyJrIjoiaFU0SUZNM25LVFlLVFBzNHo2d0dUVWk5SFQ1Wk1TNHQiLCJuIjoidGVzdDIiLCJpZCI6MX0" $str1 > ~/Desktop/GrafanaImages/p$1.png
cd ~/Desktop/GrafanaImages/

####  Added by cpf ######## 
git add p$1.png

git commit -m 'p$1'
git push origin master
