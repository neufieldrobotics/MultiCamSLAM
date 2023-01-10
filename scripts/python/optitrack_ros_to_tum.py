import rospy
import rosbag
from geometry_msgs.msg import PoseStamped
import argparse
import tf

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    
def converter(bag_file, topic_names, output_file):
    bag = rosbag.Bag(bag_file)
    #open the file
    fobj = open(output_file, "a")
    for topic, msg, t in bag.read_messages(topics=topic_names):
        tt=msg.header.stamp-rospy.Duration(0)
        s=str(format(tt.to_nsec()*1e-9, '.9f'))+" "+str(msg.pose.position.x)+" "+str(msg.pose.position.y)\
          +" "+str(msg.pose.position.z)+" "+str(msg.pose.orientation.x)+" "\
          +str(msg.pose.orientation.y) + " "+str(msg.pose.orientation.z)+" "+\
          str(msg.pose.orientation.w)+"\n"
        print("message header stamp:", format(msg.header.stamp.to_nsec()*1e-9, '.9f'))
        print("rosbag stamp:", format(t.to_nsec()*1e-9, '.9f'))
        fobj.write(s)
    fobj.close()

def converterSVO(bag_file, topic_names, output_file):
    bag = rosbag.Bag(bag_file)
    #open the file
    fobj = open(output_file, "w")
    for topic, msg, t in bag.read_messages(topics=topic_names):
        tt=msg.header.stamp-rospy.Duration(0)
        #s=str(format(tt.to_nsec()*1e-9, '.9f'))+" "+str(-1*msg.pose.position.x)+" "+str(msg.pose.position.z)\
        #  +" "+str(msg.pose.position.y)+" "+str(-1*msg.pose.orientation.x)+" "\
        #  +str(msg.pose.orientation.z) + " "+str(msg.pose.orientation.y)+" "+\
        #  str(msg.pose.orientation.w)+"\n"
        s=str(format(tt.to_nsec()*1e-9, '.9f'))+" "+str(-1*msg.pose.position.z)+" "+str(msg.pose.position.x)\
          +" "+str(-1*msg.pose.position.y)+" "+str(-1*msg.pose.orientation.z)+" "\
          +str(msg.pose.orientation.x) + " "+str(-1*msg.pose.orientation.y)+" "+\
          str(msg.pose.orientation.w)+"\n"
        print(msg.header.stamp.secs)
        print(tt)
        fobj.write(s)
    fobj.close()

def TFconverter(bag_file, topic_names, output_file):
    bag = rosbag.Bag(bag_file)
    #open the file
    fobj = open(output_file, "w")
    for topic, msg, t in bag.read_messages(topics=topic_names):
        for transf in msg.transforms:
            tt=transf.header.stamp-rospy.Duration(0)
            s=str(format(tt.to_nsec()*1e-9, '.9f'))+" "+str(-1*transf.transform.translation.x)+" "+str(transf.transform.translation.y)\
              +" "+str(transf.transform.translation.z)+" "+str(transf.transform.rotation.x)+" "\
              +str(transf.transform.rotation.y) + " "+str(transf.transform.rotation.z)+" "+\
              str(transf.transform.rotation.w)+"\n"
            print(transf.header.stamp.secs)
            print(tt)
            fobj.write(s)
    fobj.close()
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Converts the pose topics to txt files\n\n')
    parser.add_argument('-i','--input_bagfile', help='Input rosbag file to input', required=True)
    parser.add_argument('-t','--topic_name', help='topic name for conversion', required=True)
    parser.add_argument('-o','--output_file', help='Output file path', default='./output.txt')
    args=parser.parse_args()
    
    converter(args.input_bagfile, [args.topic_name] , args.output_file)
