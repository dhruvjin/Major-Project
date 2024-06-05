from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.contrib.auth.models import User
from .models import Profile
import subprocess
from datetime import datetime, timedelta
from .models import Test, Question, Answer, Profile, TestResult, CandidateTestStatus
from .forms import TestForm, QuestionForm, AnswerForm
from django.db.models import Exists, OuterRef
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from auto_interview import shared
import signal
from django.core.files.base import ContentFile
import threading as th
from django.http import FileResponse
import logging
import win32gui
import win32con
import pywin32_system32
logging.basicConfig(level=logging.INFO)

# File operation lock
file_lock = th.Lock()

head_pose_thread = None
audio_thread = None
detection_thread = None


 # Import the save_session_data function from detection.py
# Global variable to track the subprocess
process = None
stop_flag = False 
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from .models import Test, Question, Answer, Profile, TestResult, CandidateTestStatus
import subprocess,os
from datetime import datetime, timedelta

# Global variable to track the subprocess
process = None
test_idx=None
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
def chatbot_view(request):
    return render(request, 'chatbot.html')

import json

@csrf_exempt
def forward_query_to_admin(request):
    from_email = settings.EMAIL_HOST_USER
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        query = data.get('query')
        print(query)
        if query:
            
            send_mail(
                'New query from chatbot user',
                query,
                from_email, 
                ['dhruvjindal258@gmail.com'],  
                fail_silently=False,
            )
            return JsonResponse({'success': True})
    return JsonResponse({'success': False})

def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            profile = user.profile
            if profile.role == 'test_giver':
                return redirect('home_giver')
            else:
                return redirect('home_taker')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def user_signup(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password1')
        email = request.POST.get('email')
        role = request.POST.get('role')

        user = User.objects.create_user(username=username, password=password,email=email)
        profile = Profile.objects.create(user=user, role=role)
        profile.save()
        user = authenticate(username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('login')
    return render(request, 'signup.html')

def home_giver(request):
    user_name = request.user.username
    return render(request, 'home_giver.html', {'user_name': user_name})


@login_required
def home_taker(request):
    user_name = request.user.username
    candidate_profile = request.user.profile
    
    assigned_tests = candidate_profile.assigned_tests.all()
    test_statuses = CandidateTestStatus.objects.filter(candidate=candidate_profile)
    
    tests_taken_ids = test_statuses.filter(has_taken=True).values_list('test_id', flat=True)
    tests_taken = Test.objects.filter(id__in=tests_taken_ids)
    
    new_tests_assigned = assigned_tests.exclude(id__in=tests_taken_ids)
    
    return render(request, 'home_taker.html', {'user_name': user_name, 'new_tests_assigned': new_tests_assigned, 'tests_taken': tests_taken})
stop_event = th.Event()
  
def About_us(request):
    return render(request,'About.html')

def start_threads():
    global head_pose_thread, audio_thread, detection_thread
    if head_pose_thread is None or not head_pose_thread.is_alive():
        head_pose_thread = th.Thread(target=pose)
        head_pose_thread.start()
        print("Head pose thread started.")
    if audio_thread is None or not audio_thread.is_alive():
        audio_thread = th.Thread(target=sound)
        audio_thread.start()
        print("Audio thread started.")
    if detection_thread is None or not detection_thread.is_alive():
        detection_thread = th.Thread(target=run_detection)
        detection_thread.start()
        print("Detection thread started.")
@login_required
def run_face_recognition(request):
    global head_pose_thread, audio_thread, detection_thread
    
    # Clear stop event
    stop_event.clear()
    
    # Start the threads
    head_pose_thread = th.Thread(target=pose)
    audio_thread = th.Thread(target=sound)
    detection_thread = th.Thread(target=run_detection)
    
    head_pose_thread.start()
    audio_thread.start()
    detection_thread.start()
    
    # Render the response
    end_time = datetime.now() + timedelta(hours=2)
    response = render(request, 'test.html', {'end_time': end_time.isoformat()})
    return response





from django.core.management import call_command

import os
import time

def stop_threads():
    global head_pose_thread, audio_thread, detection_thread
    if head_pose_thread is not None:
        head_pose_thread.do_run = False
        head_pose_thread.join()
        head_pose_thread = None
        print("Head pose thread stopped.")
    if audio_thread is not None:
        audio_thread.do_run = False
        audio_thread.join()
        audio_thread = None
        print("Audio thread stopped.")
    if detection_thread is not None:
        detection_thread.do_run = False
        detection_thread.join()
        detection_thread = None
        print("Detection thread stopped.")

def stop_face_recognition(request):
    global head_pose_thread, audio_thread, detection_thread
    
   
    stop_event.set()
    
   
    

    stop_threads()
    return HttpResponse("Face recognition script stopped.")

def submit_test(request):
    return render(request, 'submit.html')

@login_required
def create_test(request):
    if request.method == 'POST':
        form = TestForm(request.POST)
        if form.is_valid():
            test = form.save(commit=False)
            test.created_by = request.user.profile
            test.save()
            form.save_m2m()
            
            
            candidates = test.candidates.all()
            
           
            subject = 'New Test Assigned'
            message = f'You have been assigned a new test: {test.name}. Please log in to take the test.'
            from_email = settings.EMAIL_HOST_USER
            
            
            for candidate in candidates:
                recipient_email = candidate.user.email
                send_mail(subject, message, from_email, [recipient_email])

            return redirect('add_questions', test_id=test.id)
    else:
        form = TestForm()
    return render(request, 'create_test.html', {'form': form})


@login_required
def add_questions(request, test_id):
    test = get_object_or_404(Test, id=test_id)
    if request.method == 'POST':
        form = QuestionForm(request.POST)
        if form.is_valid():
            question = form.save(commit=False)
            question.test = test
            question.save()
            return redirect('add_questions', test_id=test.id)
    else:
        form = QuestionForm()
    return render(request, 'add_questions.html', {'form': form, 'test': test})

@login_required
def start_test(request, test_id):
    global test_idx
    test_idx=test_id
    if not request.user.is_authenticated:
        return redirect('login')

    test = get_object_or_404(Test, id=test_id)
    candidate_profile = request.user.profile

   
    candidate_test_status, created = CandidateTestStatus.objects.get_or_create(candidate=candidate_profile, test=test)
    if candidate_test_status.has_taken:
        return render(request, 'test_already_completed.html')

    if request.method == 'POST':
        questions = test.questions.all()
        for question in questions:
            answer_text = request.POST.get(f'question_{question.id}')
            if answer_text:
                Answer.objects.create(
                    question=question,
                    candidate=candidate_profile,
                    text=answer_text,
                    is_correct=(answer_text.strip() == question.correct_answer.strip())
                )
        
        candidate_test_status.has_taken = True
        candidate_test_status.save()

        stop_face_recognition(request)
        test_result(request,test_id)
        return render(request, 'test_completed.html')

    else:
            stop_threads()
           
            start_threads()
            end_time = datetime.now() + timedelta(minutes=test.duration)
            duration_seconds = test.duration * 60
            duration_hours = duration_seconds // 3600
            duration_minutes = (duration_seconds % 3600) // 60
            duration_seconds = duration_seconds % 60
            response= render(request, 'start_test.html', {
                'test': test,
                'end_time': end_time.isoformat(),
                'duration_hours': duration_hours,
                'duration_minutes': duration_minutes,
                'duration_seconds': duration_seconds,
                'questions': test.questions.all()
            })
            return response


def test_result(request, test_id):
    test = get_object_or_404(Test, id=test_id)
    candidate_profile = request.user.profile

    candidate_answers = Answer.objects.filter(candidate=candidate_profile, question__test=test)
    total_marks = sum(question.marks for question in test.questions.all())
    scored_marks = sum(
        question.marks for question in test.questions.filter(
            answers__in=candidate_answers, answers__is_correct=True
        )
    )

    score_percentage = (scored_marks / total_marks) * 100 if total_marks > 0 else 0

   

    cheated =read_final_result(test_id)
       
           
     

   
    
    status = 'Fail' if cheated or score_percentage < 70 else 'Pass'

    test_result, created = TestResult.objects.get_or_create(
        test=test,
        candidate=candidate_profile,
        defaults={'score': score_percentage, 'cheated': cheated, 'status': status}
    )

    if not created:
        test_result.score = score_percentage
        test_result.cheated = cheated
        test_result.status = status
        test_result.save()

    
    session_data_path = 'session_1_data.csv'
    if os.path.exists(session_data_path):
        with open(session_data_path, 'rb') as file:
            content = file.read()
            session_data_file = ContentFile(content, name=f'session_{test_result.id}_data.csv')
            test_result.session_data.save(session_data_file.name, session_data_file)
             
    return HttpResponse('Test result saved successfully.')

def view_test_results(request):
    if request.user.profile.role != 'test_giver':
        return HttpResponse("Unauthorized", status=401)

    interviewer_profile = request.user.profile

    
    test_results = TestResult.objects.filter(test__created_by=interviewer_profile).select_related('test', 'candidate')

    return render(request, 'view_test_results.html', {'test_results': test_results})
def download_session_data(request, test_result_id):
    test_result = TestResult.objects.get(id=test_result_id)
    if test_result.session_data:
        file_path = test_result.session_data.path
        return FileResponse(open(file_path, 'rb'), as_attachment=True)
    else:
        return HttpResponse("Session data not available.")


import threading as th
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import csv
import time
import sounddevice as sd
import cv2
import mediapipe as mp
import keyboard


PLOT_LENGTH = 200
GLOBAL_CHEAT = 0
PERCENTAGE_CHEAT = 0
CHEAT_THRESH = 0.6
XDATA = list(range(200))
YDATA = [0] * 200

SOUND_AMPLITUDE = 0
AUDIO_CHEAT = 0
CALLBACKS_PER_SECOND = 38
SUS_FINDING_FREQUENCY = 2
SOUND_AMPLITUDE_THRESHOLD = 20
FRAMES_COUNT = int(CALLBACKS_PER_SECOND / SUS_FINDING_FREQUENCY)
AMPLITUDE_LIST = [0] * FRAMES_COUNT
SUS_COUNT = 0
count = 0

x = 0
y = 0
X_AXIS_CHEAT = 0
Y_AXIS_CHEAT = 0



def avg(current, previous):
    if previous > 1:
        return 0.65
    if current == 0:
        if previous < 0.01:
            return 0.01
        return previous / 1.01
    if previous == 0:
        return current
    return 1 * previous + 0.1 * current


def print_sound(indata, outdata, frames, time, status):
    global SOUND_AMPLITUDE, SUS_COUNT, count, AUDIO_CHEAT
    vnorm = int(np.linalg.norm(indata) * 10)
    AMPLITUDE_LIST.append(vnorm)
    count += 1
    AMPLITUDE_LIST.pop(0)
    if count == FRAMES_COUNT:
        avg_amp = sum(AMPLITUDE_LIST) / FRAMES_COUNT
        SOUND_AMPLITUDE = avg_amp
        if SUS_COUNT >= 2:
            AUDIO_CHEAT = 1
            SUS_COUNT = 0
        if avg_amp > SOUND_AMPLITUDE_THRESHOLD:
            SUS_COUNT += 1
        else:
            SUS_COUNT = 0
            AUDIO_CHEAT = 0
        count = 0

def sound():
    with sd.Stream(callback=print_sound):
        while not stop_event.is_set():  
            sd.sleep(100)


import cv2
import ctypes


def moveWindowTo(x, y):
    ctypes.windll.user32.SetWindowPos(
        ctypes.windll.user32.FindWindowW(None, "Head Pose Estimation"),
        0,
        x,
        y,
        0,
        0,
        0x0001
    )


def hideTitleBar(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
    style &= ~win32con.WS_CAPTION
    style &= ~win32con.WS_THICKFRAME
    win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)
    win32gui.SetWindowPos(hwnd, None, 0, 0, 0, 0,
                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOZORDER | win32con.SWP_FRAMECHANGED)


def moveWindowTo(x, y):
    ctypes.windll.user32.SetWindowPos(
        ctypes.windll.user32.FindWindowW(None, "Head Pose Estimation"),
        0,
        x,
        y-45,
        0,
        0,
        0x0001
    )
def set_fixed_size(window_name, width, height):
    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
    style = ctypes.windll.user32.GetWindowLongW(hwnd, -16)  
    style &= ~0x00040000  
    style &= ~0x00020000  
    ctypes.windll.user32.SetWindowLongW(hwnd, -16, style)
    ctypes.windll.user32.SetWindowPos(hwnd, None, 0, 0, width, height, 0x0002)

def pose():
    global x, y, X_AXIS_CHEAT, Y_AXIS_CHEAT
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils
    face_ids = [33, 263, 1, 61, 291, 199]

    window_name = "Head Pose Estimation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    
    screen_width = ctypes.windll.user32.GetSystemMetrics(0)
    screen_height = ctypes.windll.user32.GetSystemMetrics(1)
    window_width = 440
    window_height = 340
    cv2.resizeWindow(window_name, window_width, window_height)
    moveWindowTo(screen_width - window_width - 10, 10)

    
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    set_fixed_size(window_name, window_width, window_height)
   
    while cap.isOpened() and not stop_event.is_set():  
        success, image = cap.read()
        if not success:
            break
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None)
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in face_ids:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                x = angles[0] * 360
                y = angles[1] * 360

                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                else:
                    text = "Forward"
                text = str(int(x)) + "::" + str(int(y)) + text

                if y < -10 or y > 10:
                    X_AXIS_CHEAT = 1
                else:
                    X_AXIS_CHEAT = 0

                if x < -5:
                    Y_AXIS_CHEAT = 1
                else:
                    Y_AXIS_CHEAT = 0

                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                cv2.line(image, p1, p2, (255, 0, 0), 2)
                cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(window_name, image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def process():
    global GLOBAL_CHEAT, PERCENTAGE_CHEAT
    if GLOBAL_CHEAT == 0:
        if X_AXIS_CHEAT == 0:
            if Y_AXIS_CHEAT == 0:
                if AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.2, PERCENTAGE_CHEAT)
            else:
                if AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.2, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.4, PERCENTAGE_CHEAT)
        else:
            if Y_AXIS_CHEAT == 0:
                if AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.1, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.4, PERCENTAGE_CHEAT)
            else:
                if AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.15, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.25, PERCENTAGE_CHEAT)
    else:
        if X_AXIS_CHEAT == 0:
            if Y_AXIS_CHEAT == 0:
                if AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.55, PERCENTAGE_CHEAT)
            else:
                if AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.55, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.85, PERCENTAGE_CHEAT)
        else:
            if Y_AXIS_CHEAT == 0:
                if AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.6, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.85, PERCENTAGE_CHEAT)
            else:
                if AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.5, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.85, PERCENTAGE_CHEAT)

    if PERCENTAGE_CHEAT > CHEAT_THRESH:
        GLOBAL_CHEAT = 1
        print("CHEATING")
    else:
        GLOBAL_CHEAT = 0
    print("Cheat percent: ", PERCENTAGE_CHEAT, GLOBAL_CHEAT)


def save_final_result(session_number, cheated):
    global test_idx
    try:
        with file_lock:
            with open(f'final_results_{test_idx}.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Cheated' if cheated else 'Not'])
                file.flush()  
        time.sleep(0.1)  
        logging.info(f'Successfully wrote final results for test {test_idx}')
    except Exception as e:
        logging.error(f'Error writing final results for test {test_idx}: {e}')


def read_final_result(test_idx):
    cheated = False
    final_results_path = f'final_results_{test_idx}.csv'
    if os.path.exists(final_results_path):
        try:
            with file_lock:
                with open(final_results_path, 'r') as file:
                    content = file.read().strip()
                    logging.info(f'Content read from final results file for test {test_idx}: "{content}"')
                    if 'Cheated' in content:
                        cheated = True
            logging.info(f'Successfully read final results for test {test_idx}')
        except Exception as e:
            logging.error(f'Error reading final results for test {test_idx}: {e}')
    return cheated

def save_session_data(session_number, data, cheated):
    with open(f'session_{session_number}_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Cheat Probability'])
        for entry in data:
            writer.writerow(entry)



def run_detection():
    global XDATA, YDATA
    #plt.ion()  # Turn on interactive mode
    #plt.figure()  # Create a new figure
    axes = plt.gca()
    axes.set_xlim(0, 200)
    axes.set_ylim(0, 1)
    line, = axes.plot(XDATA, YDATA, 'r-')
    plt.title("Suspicious Behaviour Detection")
    plt.xlabel("Time")
    plt.ylabel("Cheat Probability")
    session_number = 1
    session_data = []
    i = 0
    exceed_count = 0
    iterations_without_cheat = 0
    cheating_occurred = False  

    while not stop_event.is_set():  
        if keyboard.is_pressed('q'):  
            break

        YDATA.pop(0)
        YDATA.append(PERCENTAGE_CHEAT)
        line.set_xdata(XDATA)
        line.set_ydata(YDATA)
        #plt.draw()
        #plt.pause(1e-17)
        time.sleep(1/5)
        process()
        session_data.append([i, PERCENTAGE_CHEAT])
        if PERCENTAGE_CHEAT > CHEAT_THRESH:
            exceed_count += 1
            if exceed_count >= 10:
                cheating_occurred = True
        else:
            iterations_without_cheat += 1

        if i % PLOT_LENGTH == 0 and i != 0:
            break
        i += 1

    save_session_data(session_number, session_data, cheating_occurred)

    if cheating_occurred:
        save_final_result(session_number, True)
    else:
        save_final_result(session_number, False)

   # plt.close()




