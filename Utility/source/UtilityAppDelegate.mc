import Toybox.Lang;
import Toybox.WatchUi;
import Toybox.Application.Storage;

class UtilityAppDelegate extends WatchUi.BehaviorDelegate {

    function initialize() {
        BehaviorDelegate.initialize();
    }

    function onMenu() as Boolean 
    {
        var menu = new WatchUi.Menu();
        var delegate = new UtilityAppMenuDelegate();

        var app = getApp();

        if (app._isRecording)
        {
            menu.addItem("Stop recording", "stop");
            menu.addItem("Cancel", "cancel");
        }
        else
        {
            menu.addItem("Start recording", "start");

            var keys = Storage.getValue("keys");

            if (keys != null)
            {
                menu.addItem("Send data", "send");
                menu.addItem("Clear data", "clear");
            }
        }

        WatchUi.pushView(menu, delegate, WatchUi.SLIDE_DOWN);
        return true;
    }
}