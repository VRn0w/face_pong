var video = document.querySelector("#live");
 
navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia || navigator.oGetUserMedia;
 
if (navigator.getUserMedia) {       
    navigator.getUserMedia({video: true}, handleVideo, videoError);
}
 
var video_initialized = false;

function handleVideo(stream) {
    video.src = window.URL.createObjectURL(stream);
    video_initialized = true;
    //alert('video initialized.');
}
 
function videoError(e) {
    // do something
}

function dataURItoBlob(dataURI) {
    var binary = atob(dataURI.split(',')[1]);
    var array = [];
    for(var i = 0; i < binary.length; i++) {
        array.push(binary.charCodeAt(i));
    }
    return new Blob([new Uint8Array(array)], {type: 'image/jpeg'});
}

var canvas = document.querySelector('#canvas');
var ctx = canvas.getContext('2d');


var response_canvas = document.querySelector('#response_canvas');
var response_ctx = response_canvas.getContext('2d');


var ws = new WebSocket("ws://localhost:8888/websocket");

ws.onopen = function(){
    
    timer = setInterval(
        function () {
            if(video_initialized == true){
                ctx.drawImage(video, 0, 0, 320, 240);
                //ws.send(canvas.toDataURL('image/jpeg',1.0));
                ws.send(canvas.toDataURL());
                //var data = canvas.toDataURL('image/jpeg', 1.0);
                //newblob = dataURItoBlob(data);
                //ws.send(newblob);
            }         
        }, 30);
}
ws.onmessage = function(evt){
    var img = new Image();
    img.src = "data:image/png;base64,"+evt.data;
    img.onload = function () {
        response_ctx.drawImage(img,0,0, 320,240);
    }
    //document.getElementById("debugHTML").textContent=document.getElementById("debugHTML").textContent + "<br> newImage";
}
    
