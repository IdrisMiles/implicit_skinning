#ifndef IMPLICITSKINSETTINGS_H
#define IMPLICITSKINSETTINGS_H

//-------------------------------------------------------------------------------

#include <QGroupBox>
#include <QPushButton>
#include <QLineEdit>
#include <QSpinBox>
#include <QDoubleSpinBox>

//-------------------------------------------------------------------------------

namespace Ui {
class ImplicitSkinSettings;
}

//-------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 18/04/2017
//-------------------------------------------------------------------------------


/// @class ImplicitSkinSettings
/// @brief This class inherits from QGroupBox, it is a widget for selecting an animation file to load
/// as well as editing settings for implicit skinning.
class ImplicitSkinSettings : public QGroupBox
{
    Q_OBJECT

public:
    /// @brief constructor
    explicit ImplicitSkinSettings(QWidget *parent = 0);

    /// @brief destructor
    ~ImplicitSkinSettings();

    /// @brief Method to get the animation file that has been selected
    std::string GetAnimationFile();

    /// @brief Method to get the number of iterations for implicit skinning
    int GetIterations();

    /// @brief Method to get the value of sigma for implicit skinning
    double GetSigma();

    /// @brief Method to get the contact angle used for imiplicit skinning
    double GetContactAngle();

signals:
    /// @brief Qt Signal emitted when iteration changed.
    void IterationsChanged(int iterations);

    /// @brief Qt Signal emitted when sigma chanegd
    void SigmaChanged(double sigma);

    /// @brief Qt Signal emitted when contact angle changed
    void ContactAngleChanged(double contactAngle);

    /// @brief Qt Signal emitted when load animation button clicked
    void LoadAnimationClicked();

    /// @brief Qt Signal emitted when browse animation button clicked
    void BrowseAnimationClicked();

    /// @brief Qt Signal emitted when animation file name changed
    void AnimationFileChnaged(std::string file);

public slots:


private:
    /// @brief ui widgets
    Ui::ImplicitSkinSettings *ui;
};

//-------------------------------------------------------------------------------

#endif // IMPLICITSKINSETTINGS_H
