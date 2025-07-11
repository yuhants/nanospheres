import time
import takedata
import movemagnet
import control_hugo as ch

stepsize_up = [500, 3500]
stepsize_down = [490, 3430]

numsteps_up = 2
numsteps_down = 2

magnet = movemagnet.StepperMotorMag(port = 'COM3')

AppName      = 'lock_in+pid'
host         = '169.254.208.142'
port         = 22  # default port

rpPID =ch.red_pitaya_app(AppName=AppName,host=host,port=port,password='root', filename = None) # This is so we can reset the homodyne feedback after magnet move

uds = 0

while uds < 10:
    for i in range(numsteps_up):
        if i == 0:
            takedata.main(file_directory = rf"E:\lorentz_force\sphere20250604\20250610\3 V_flip\pos"+str(i)+" 1" + str(uds))
        magnet.not_sleep()
        magnet.move_up(stepsize_up[i])
        time.sleep(1)
        magnet.sleep()

        # Resest PID intergral
        rpPID.lock.pidB_ctrl = 1
        time.sleep(1)
        rpPID.lock.pidB_ctrl = 0
        time.sleep(1)

        #Take data
        takedata.main(file_directory = rf"E:\lorentz_force\sphere20250604\20250610\3 V_flip\pos"+str(i+1)+" 1" + str(uds))

    #input('Hit space after flipping')

    for i in range(numsteps_down):
        if i == 0:
            takedata.main(file_directory = rf"E:\lorentz_force\sphere20250604\20250610\3 V_flip\pos"+str(numsteps_down)+" 2" + str(uds))
        magnet.not_sleep()
        magnet.move_down(stepsize_down[numsteps_down-1-i])
        time.sleep(10)
        magnet.sleep()

        # Resest PID intergral
        rpPID.lock.pidB_ctrl = 1
        time.sleep(1)
        rpPID.lock.pidB_ctrl = 0
        time.sleep(1)

        #Take data
        takedata.main(file_directory = rf"E:\lorentz_force\sphere20250604\20250610\3 V_flip\pos"+str(numsteps_down-1 - i)+" 2" + str(uds))
    
    uds += 1

magnet.close()