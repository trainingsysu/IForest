# Commands:
#   hubot showImg - Reply with p1.png!

child_process = require('child_process')
module.exports = (robot) ->
  robot.hear /showImg/i, (msg) ->	
    count = Math.ceil(Math.random()*1000000)
    comd = '~/Desktop/GrafanaImages/x' + ' ' + count  
    child_process.exec comd, (err, stdout, stderr) ->
      if err
        msg.send 'ERROR!'
      else
        sleep 5000
        msg.send ' https://github.com/trainingsysu/GrafanaImages/blob/master/p' + count + '.png'

sleep = (ms)->
  start = new Date().getTime()
  continue while new Date().getTime() - start < ms
 
