import Toybox.Lang;
import Toybox.System;
import Toybox.WatchUi;
import Toybox.Application.Storage;

class UtilityAppMenuDelegate extends WatchUi.MenuInputDelegate {

    function initialize() {
        MenuInputDelegate.initialize();
    }

    function onMenuItem(item as Lang.String) as Void {

        var app = getApp();

        if (item.equals("start")) 
        {
            app.start();
        } 

        if (item.equals("stop"))
        {
            app.stop();
        }

        if (item.equals("send"))
        {
            app.send();
        }

        if (item.equals("clear"))
        {
            app.clear();
        }
    }
}