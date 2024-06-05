from django.contrib import admin
from django.urls import path
from auto_interview import views
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from auto_interview import consumers  # Replace 'auto_interview' with your app name
from django.urls import re_path

urlpatterns = [
    path('About/',views.About_us,name='About'),
    path('forward_query_to_admin/', views.forward_query_to_admin, name='forward_query_to_admin'),
    path('', views.user_login, name='login'),
    path('chatbot/',views.chatbot_view, name='chatbot'),
    path('admin/', admin.site.urls),
    path('login/', views.user_login, name='login'),
    path('signup/', views.user_signup, name='signup'),
    path('home_giver/', views.home_giver, name='home_giver'),
    path('home_taker/', views.home_taker, name='home_taker'),
    path('test/', views.run_face_recognition, name='run_face_recognition'),
    path('result/', views.stop_face_recognition, name='stop_face_recognition'),
    path('submit/', views.submit_test, name='submit_test'),
    path('create_test/', views.create_test, name='create_test'),
    path('add_questions/<int:test_id>/', views.add_questions, name='add_questions'),
    path('start_test/<int:test_id>/', views.start_test, name='start_test'),
    path('test_result/<int:test_id>/', views.test_result, name='test_result'),
    path('view_test_results/', views.view_test_results, name='view_test_results'),
    path('forward_query_to_admin/', views.forward_query_to_admin, name='forward_query_to_admin'),
    path('download-session-data/<int:test_result_id>/', views.download_session_data, name='download_session_data'),
]


websocket_urlpatterns = [
    re_path(r'ws/video_feed/$', consumers.VideoFeedConsumer.as_asgi()),
]


application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": URLRouter(websocket_urlpatterns),
})
