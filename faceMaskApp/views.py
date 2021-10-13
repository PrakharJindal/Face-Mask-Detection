from django.shortcuts import render, redirect
from django.conf import settings
from django.http.response import HttpResponseServerError, StreamingHttpResponse
from .camera import VideoCamera

regex = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'


def index(request):
    return render(request, 'core/index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()), content_type="multipart/x-mixed-replace;boundary=frame")
    # return StreamingHttpResponse(gen(VideoCamera()),
    #                              content_type='multipart/x-mixed-replace; boundary-frame')
